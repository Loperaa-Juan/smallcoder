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

    def _load_generator(model_id, device_map):
        """Load model without using transformers.pipeline.

        Priority:
        - PEFT adapter (local or remote) if peft available
        - AutoModelForCausalLM + AutoTokenizer (with trust_remote_code retry)
        """
        # Try PEFT adapter first when available
        if PeftConfig is not None and PeftModel is not None:
            try:
                peft_config = PeftConfig.from_pretrained(model_id)
                tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path, trust_remote_code=True)
                base_model = AutoModelForCausalLM.from_pretrained(
                    peft_config.base_model_name_or_path,
                    torch_dtype=torch.float32,
                    device_map=device_map,
                    trust_remote_code=True,
                )
                model = PeftModel.from_pretrained(base_model, model_id)
                return model, tokenizer, False
            except Exception as peft_exc:
                print(f'PEFT load failed: {peft_exc}', file=sys.stderr)

        # Try loading base model directly
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                device_map=device_map,
                trust_remote_code=False,
            )
            return model, tokenizer, False
        except Exception as exc:
            print(f'Base model load failed without trust_remote_code: {exc}', file=sys.stderr)

        # Retry with trust_remote_code when model requires remote code
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                device_map=device_map,
                trust_remote_code=True,
            )
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
