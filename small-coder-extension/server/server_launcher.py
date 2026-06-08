#!/usr/bin/env python3
import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftConfig, PeftModel
except ImportError:
    PeftConfig = None
    PeftModel = None

class RequestHandler(BaseHTTPRequestHandler):
    def _send_json(self, payload, status=200):
        payload_bytes = json.dumps(payload).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(payload_bytes)))
        self.end_headers()
        self.wfile.write(payload_bytes)

    def do_GET(self):
        if self.path == '/health':
            self._send_json({'status': 'ok', 'device': self.server.device, 'model': self.server.model_id})
            return
        self.send_error(404, 'Not Found')

    def do_POST(self):
        if self.path != '/predict':
            self.send_error(404, 'Not Found')
            return

        content_length = int(self.headers.get('Content-Length', 0))
        raw_body = self.rfile.read(content_length)
        try:
            body = json.loads(raw_body)
        except json.JSONDecodeError:
            self.send_error(400, 'Invalid JSON')
            return

        prompt = body.get('prompt', '')
        max_tokens = int(body.get('max_tokens', 64))
        if not isinstance(prompt, str):
            self.send_error(400, 'Invalid prompt')
            return

        try:
            completion = self._generate_completion(prompt, max_tokens)
            self._send_json({'completion': completion, 'finish_reason': 'length'})
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.send_error(500, f'Inference error: {exc}')

    def _generate_completion(self, prompt, max_tokens):
        if not prompt:
            return '\n# smallcoder placeholder completion'

        if getattr(self.server, 'use_pipeline', False):
            return self._generate_with_pipeline(prompt, max_tokens)

        return self._generate_with_model(prompt, max_tokens)

    def _generate_with_pipeline(self, prompt, max_tokens):
        generator = self.server.generator
        try:
            result = generator(
                [{'role': 'user', 'content': prompt}],
                max_new_tokens=max_tokens,
                return_full_text=False,
            )
        except Exception:
            result = generator(prompt, max_new_tokens=max_tokens, return_full_text=False)

        if isinstance(result, list) and result:
            first = result[0]
            if isinstance(first, dict) and 'generated_text' in first:
                return first['generated_text']
            if isinstance(first, dict) and 'text' in first:
                return first['text']
        if isinstance(result, dict):
            return result.get('generated_text') or result.get('text') or ''
        return str(result)

    def _generate_with_model(self, prompt, max_tokens):
        model = self.server.generator
        tokenizer = self.server.tokenizer

        if tokenizer is None:
            raise RuntimeError('Tokenizer missing for manual model generation')

        if hasattr(tokenizer, 'apply_chat_template'):
            inputs = tokenizer.apply_chat_template(
                [{'role': 'user', 'content': prompt}],
                return_tensors='pt',
                add_generation_prompt=True,
            )
            # Convert BatchEncoding (UserDict) to plain dict so **inputs works correctly
            if hasattr(inputs, 'items'):
                inputs = dict(inputs)
            else:
                inputs = {'input_ids': inputs}
        else:
            inputs = tokenizer(prompt, return_tensors='pt')

        if hasattr(inputs, 'to'):
            inputs = inputs.to(self.server.torch_device)
        else:
            for key, value in inputs.items():
                if hasattr(value, 'to'):
                    inputs[key] = value.to(self.server.torch_device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                num_beams=1,
            )

        if hasattr(inputs, 'get') and inputs.get('input_ids') is not None:
            start = inputs['input_ids'].shape[-1]
            return tokenizer.decode(output[0][start:], skip_special_tokens=True)

        return tokenizer.decode(output[0], skip_special_tokens=True)

    def log_message(self, format, *args):
        return

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SmallCoder local inference server')
    parser.add_argument('--port', type=int, default=53802)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--model-id', type=str, default='smallcoder-tiny')
    args = parser.parse_args()

    device_map = 0 if args.device == 'cuda' else "cpu"
    torch_device = torch.device('cuda' if args.device == 'cuda' else 'cpu')
    print(f'Loading model {args.model_id} on device {args.device}...')

    def _resolve_model_path(model_id):
        """Resolve a local model path, handling HF cache directory structures."""
        path = Path(model_id)
        if not path.is_dir():
            return model_id
        if (path / 'tokenizer_config.json').exists() or (path / 'config.json').exists():
            return str(path)
        # HF cache structure: models--org--repo/snapshots/<hash>/
        for models_dir in path.glob('models--*'):
            snapshots_dir = models_dir / 'snapshots'
            if snapshots_dir.is_dir():
                for snapshot in snapshots_dir.iterdir():
                    if snapshot.is_dir() and any(
                        (snapshot / f).exists()
                        for f in ('tokenizer_config.json', 'config.json', 'adapter_config.json')
                    ):
                        return str(snapshot)
        return model_id

    def _build_quant_config(device_map):
        """Build a 4-bit bitsandbytes quantization config when running on GPU.

        Returns None on CPU or when bitsandbytes / BitsAndBytesConfig are
        unavailable, so the model falls back to a normal (unquantized) load.
        """
        if device_map == 'cpu':
            return None
        try:
            from transformers import BitsAndBytesConfig
        except ImportError:
            print('BitsAndBytesConfig unavailable; loading without 4-bit quantization', file=sys.stderr)
            return None
        try:
            import bitsandbytes  # noqa: F401
        except ImportError:
            print('bitsandbytes not installed; loading without 4-bit quantization', file=sys.stderr)
            return None
        print('GPU detected: loading model in 4-bit (nf4) with bitsandbytes')
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    def _load_generator(model_id, device_map):
        """Load model without using transformers.pipeline.

        Priority:
        - PEFT adapter (local or remote) if peft available
        - AutoModelForCausalLM + AutoTokenizer (with trust_remote_code retry)
        """
        resolved_id = _resolve_model_path(model_id)
        print(f'Resolved model path: {resolved_id}')

        quant_config = _build_quant_config(device_map)

        # Try PEFT adapter first when available
        if PeftConfig is not None and PeftModel is not None:
            try:
                peft_config = PeftConfig.from_pretrained(resolved_id)
                base_id = peft_config.base_model_name_or_path
                # Prefer tokenizer bundled in the adapter snapshot over downloading from base model
                try:
                    tokenizer = AutoTokenizer.from_pretrained(resolved_id, trust_remote_code=True)
                except Exception:
                    tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
                torch_dtype = torch.float16 if device_map != 'cpu' else torch.float32
                print(f'Loading base model {base_id} (this may download ~3 GB on first run)...')
                base_kwargs = dict(
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                    trust_remote_code=True,
                )
                if quant_config is not None:
                    base_kwargs['quantization_config'] = quant_config
                base_model = AutoModelForCausalLM.from_pretrained(base_id, **base_kwargs)
                model = PeftModel.from_pretrained(base_model, resolved_id)
                # Merge LoRA weights into the base model to drop the PEFT wrapper
                # overhead on every forward pass. Skip merging on 4-bit (bitsandbytes)
                # loads: merging into quantized linears introduces rounding errors
                # that can change generations, so we keep the adapter unmerged there.
                if quant_config is None:
                    try:
                        model = model.merge_and_unload()
                        print('LoRA weights merged into base model')
                    except Exception as merge_exc:
                        print(f'merge_and_unload failed, keeping adapter unmerged: {merge_exc}', file=sys.stderr)
                else:
                    print('4-bit load detected; keeping LoRA adapter unmerged to avoid rounding errors')
                return model, tokenizer, False
            except Exception as peft_exc:
                print(f'PEFT load failed: {peft_exc}', file=sys.stderr)

        # Try loading base model directly
        try:
            tokenizer = AutoTokenizer.from_pretrained(resolved_id, trust_remote_code=False)
            base_kwargs = dict(
                torch_dtype=torch.float32,
                device_map=device_map,
                trust_remote_code=False,
            )
            if quant_config is not None:
                base_kwargs['quantization_config'] = quant_config
            model = AutoModelForCausalLM.from_pretrained(resolved_id, **base_kwargs)
            return model, tokenizer, False
        except Exception as exc:
            print(f'Base model load failed without trust_remote_code: {exc}', file=sys.stderr)

        # Retry with trust_remote_code when model requires remote code
        try:
            tokenizer = AutoTokenizer.from_pretrained(resolved_id, trust_remote_code=True)
            base_kwargs = dict(
                torch_dtype=torch.float32,
                device_map=device_map,
                trust_remote_code=True,
            )
            if quant_config is not None:
                base_kwargs['quantization_config'] = quant_config
            model = AutoModelForCausalLM.from_pretrained(resolved_id, **base_kwargs)
            return model, tokenizer, False
        except Exception as exc2:
            print(f'Base model load failed with trust_remote_code: {exc2}', file=sys.stderr)
            raise RuntimeError(f'Unable to load model {model_id}: {exc2}')

    generator, tokenizer, use_pipeline = _load_generator(args.model_id, device_map)

    server = ThreadedHTTPServer(('127.0.0.1', args.port), RequestHandler)
    server.device = args.device
    server.model_id = args.model_id
    server.generator = generator
    server.tokenizer = tokenizer
    server.use_pipeline = use_pipeline
    server.torch_device = torch_device

    print(f'SmallCoder server starting on http://127.0.0.1:{args.port} device={args.device} model={args.model_id}')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
