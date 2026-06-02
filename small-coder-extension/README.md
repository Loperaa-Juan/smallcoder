# SmallCoder VS Code Extension

This extension provides a local SmallCoder runtime integration for VS Code. It supports:

- local virtual environment setup
- a minimal local Python server for inference
- an editor command to predict text from the current cursor position
- runtime log viewing

## Commands

- `SmallCoder: Setup runtime` — create a Python venv and prepare the local server
- `SmallCoder: Predict from cursor` — send current editor context to the local server and insert completion
- `SmallCoder: Toggle device (CPU / CUDA)` — change runtime device preference
- `SmallCoder: Download model` — prepare model metadata in extension storage
- `SmallCoder: Open runtime logs` — open the SmallCoder output channel

## Runtime files

- `src/extension.ts` — activation and command registration
- `src/ui.ts` — user interaction helpers
- `src/envManager.ts` — virtual environment creation and pip install
- `src/modelManager.ts` — model selection and download scaffolding
- `src/serverManager.ts` — local Python server process management
- `src/predict.ts` — editor context capture and prediction insertion
- `server/server_launcher.py` — minimal local HTTP inference server

## Notes

- The current server implementation is a stub server compatible with the local runtime flow.
- The extension stores runtime state in VS Code global state and uses the configured `smallCoder.device` setting.
