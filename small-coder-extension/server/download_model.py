#!/usr/bin/env python3
"""Download a model from HuggingFace Hub and save to local path."""
import argparse
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError as exc:
    print(f'Missing dependency: huggingface_hub. Install with `pip install huggingface-hub`.', file=sys.stderr)
    raise exc


def download_model(model_id: str, cache_dir: str) -> str:
    """Download model from HuggingFace Hub.
    
    Args:
        model_id: HuggingFace model identifier (e.g., 'Juanxxo/qwen2.5-1.5B-code-adapter')
        cache_dir: Local directory to store model files
        
    Returns:
        Path to downloaded model directory
    """
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    print(f'Downloading model {model_id}...')
    
    # Download model and return local path
    model_path = snapshot_download(
        repo_id=model_id,
        cache_dir=cache_dir,
        allow_patterns=['*.json', '*.bin', '*.safetensors', '*.model', '*.pt'],
    )
    
    print(f'Model downloaded to {model_path}')
    return model_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download HuggingFace model')
    parser.add_argument('--model-id', type=str, required=True, help='Model ID (e.g., Juanxxo/qwen2.5-1.5B-code-adapter)')
    parser.add_argument('--cache-dir', type=str, required=True, help='Local directory to store model')
    args = parser.parse_args()
    
    try:
        path = download_model(args.model_id, args.cache_dir)
        print(path)  # Output path to stdout for parent process
    except Exception as exc:
        print(f'Download failed: {exc}', file=sys.stderr)
        sys.exit(1)
