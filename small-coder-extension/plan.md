# small-coder-extension — Plan

## Overview
small-coder-extension is a VS Code extension that on activation:
- Shows the model(s) available for download and lets the user choose one.
- Creates a Python virtual environment appropriate for the user's OS.
- Adds a Command Palette command: "Predict text from cursor" that autocompletes the current line(s).
- Uses CPU by default; offers an option to use NVIDIA GPU (when available).
- Includes scaffolding and optional support for referencing an external sourcemap for implementation patterns.

## Goals & scope
- Minimal, secure local setup flow that does not upload code.
- Simple, discoverable UX: activation prompt -> venv creation -> model download -> usage via command palette and keybinding.
- Model runtime is local (Python-backed) and CPU-first; optional CUDA path for NVIDIA.
- Support Windows, macOS, Linux; detect OS and choose venv strategy accordingly.

## Project structure (scaffold)
- package.json (VS Code extension manifest)
- src/
  - extension.ts (activation, commands registration)
  - ui.ts (UI helpers: quickPick, notifications, progress)
  - envManager.ts (create/manage virtualenvs per OS)
  - modelManager.ts (list/download/verify models)
  - server/
    - server_launcher.py (starts minimal local Python HTTP/RPC server to run model inference)
    - requirements.txt (runtime python deps)
  - predict.ts (command implementation: sends cursor context to server and inserts completion)
  - telemetry.ts (optional, minimal and explicit opt-in)
- scripts/
  - build, test helpers
- README.md, plan.md (this file), LICENSE
- .vscode/ (recommended settings + keybinding for "Predict text from cursor")

## Activation flow
1. On extension activation (first run or when no runtime configured):
   - Show a QuickPick: "Choose model to download" with available models (small/medium/fast).
   - Show estimated disk size and default runtime (CPU).
2. After model selection:
   - Prompt to create virtual environment: "Create virtual environment at <path>?" with Edit and Confirm.
   - Detect OS and propose the venv command:
     - Linux/macOS: python3 -m venv <path>
     - Windows: python -m venv <path> (or use py -3 -m venv)
   - Offer an advanced option to use conda if user prefers (optional).
3. Create venv, install server requirements (pip install -r requirements.txt).
   - Show progress and errors; provide a "Retry" and "Open logs" action on failure.
4. Download the selected model into an extension-managed storage folder (or user path).
   - Verify checksum if available.
5. Launch the Python server (background process) with chosen device option (cpu or cuda).
6. Register/enable the "Predict text from cursor" command.

## Commands & UX
- Command Palette:
  - "small-coder: Setup runtime" — rerun setup (choose model/venv/device).
  - "small-coder: Predict text from cursor" — send context and insert prediction.
  - "small-coder: Toggle device (CPU / NVIDIA CUDA)" — toggle runtime preference (restarts server).
  - "small-coder: Download model" — download/replace model manually.
  - "small-coder: Open runtime logs"
- Editor context:
  - Optional keybinding (e.g., Ctrl+Alt+P) for "Predict text from cursor".
- QuickPick/Modal UX:
  - Keep prompts minimal, show progress notifications and non-blocking background tasks.
  - Provide clear error messages and recovery steps (retry, open terminal, view logs).

## Virtual environment management (envManager.ts)
- Detect Python availability:
  - On macOS/Linux: prefer python3, fallback to python.
  - On Windows: try py -3 then python.
- venv creation options:
  - Use built-in venv: run <python> -m venv <venvPath>
  - Activate and pip install in a child process:
    - Linux/macOS:
      - source <venvPath>/bin/activate && pip install -r requirements.txt
    - Windows PowerShell:
      - <venvPath>\Scripts\Activate.ps1; pip install -r requirements.txt
    - Windows CMD:
      - <venvPath>\Scripts\activate.bat && pip install -r requirements.txt
- Use spawn with cross-platform shell flags and capture stdout/stderr to extension logs.
- Store venv path in extension globalState and show in settings.

## Model management (modelManager.ts)
- Offer a small curated list of model options (names, sizes, download URLs, checksums).
- Download with progress (streaming) and write to disk; allow user to cancel.
- Verify integrity (sha256) if checksum present.
- Expose model path and meta to server_launcher.py via config or CLI args.

## Local Python server (server_launcher.py)
- Minimal HTTP or JSON-RPC server (e.g., Flask or FastAPI) that:
  - Loads the chosen model with CPU default; if device == "cuda" and GPU available, use CUDA device.
  - Exposes an endpoint POST /predict with payload: { "context": "...", "cursor_pos": N, "max_tokens": M }
  - Returns completion text and metadata (logprob optional).
- Requirements: lightweight inference libs (e.g., transformers + accelerate, or a small local model runtime). Keep requirements.txt versioned.
- Health endpoint /health for readiness.
- Server should bind to localhost on a random free port and output port to stdout; extension reads it to connect.

Security note: run only on localhost; ensure no external network binding.

## Predict command (predict.ts)
- Gather editor state:
  - Current document text and cursor position (or selection).
  - Send minimal context (e.g., current line, previous few lines, file language) to server.
- Show a quick inline spinner while awaiting a response.
- On receive: insert or replace appropriate text at cursor; support an undo step.
- Allow user to accept, reject, or request more tokens (via subsequent command invocation).

## Device selection & runtime
- Default: cpu
- Option in settings and a toggle command to request NVIDIA (cuda) device.
- When user toggles device:
  - Stop server and relaunch with device arg.
  - If CUDA not available or runtime GPU packages missing, show clear error and fallback to CPU.
- If GPU device selected, provide notes on required drivers (NVIDIA driver + CUDA) and optional Conda environment path if helpful.

## Settings (package.json contributes.configuration)
- smallCoder.model (string): chosen model id
- smallCoder.envPath (string): path to venv
- smallCoder.device (enum): "cpu" | "cuda"
- smallCoder.maxTokens (number): default tokens for prediction
- smallCoder.serverArgs (object): extra args passed to server launcher
- smallCoder.autoCreateVenv (boolean): whether to auto-create venv on activation

## Logging & diagnostics
- Capture child process stdout/stderr to an extension output channel "small-coder".
- Save installation logs to extension storage for troubleshooting.
- Expose "Open runtime logs" command to display logs.
- Surface errors via VS Code notifications with actionable buttons.

## Tests & CI
- Unit tests for:
  - envManager: simulate OS cases and ensure correct commands chosen.
  - modelManager: mock download and checksum verification.
  - predict.ts: simulate server responses and ensure edits applied safely.
- Integration test:
  - Launch the server in a CI container (CPU only) and run a simple predict call.
- Publish with GitHub Actions: build, lint, run tests, package vsix.

## Implementation notes & best practices
- Keep the extension lightweight: heavy inference runs are done in the Python server, not the extension host.
- Use background processes and non-blocking UI to avoid freezing VS Code.
- Follow VS Code extension security: use localhost only, avoid elevated privileges.
- Provide a single-file minimal example server_launcher.py in the repo to lower barrier.
- Provide fallback behavior if Python or pip not available: show clear instructions and links to install Python.

## Example minimal server contract
- POST /predict
  - Request: { "prompt": "text before cursor", "max_tokens": 64, "language": "python" }
  - Response: { "completion": "predicted text", "finish_reason": "length" }
- GET /health
  - Response: { "status": "ok", "device": "cpu", "model": "small-model-v1" }

## Migration & reference sourcemap
- If you provide a sourcemap of another application, map these parts:
  - UI prompts -> ui.ts
  - venv & subprocess handling -> envManager.ts
  - model download and storage -> modelManager.ts
  - server start/stop and IPC -> server_launcher.py + extension process management
  - command registration and editor edits -> extension.ts + predict.ts
- Use the sourcemap to replicate patterns for logging, error handling, and cross-process communication.

## Milestones (suggested)
1. Scaffold extension, register commands, basic UI QuickPick (1–2 days)
2. Implement env creation (cross-platform) and logging (2–3 days)
3. Implement model download and verification (2–3 days)
4. Add minimal Python server and server lifecycle management (3–5 days)
5. Implement "Predict text from cursor" end-to-end (2–3 days)
6. Device toggle, settings, error handling, tests, docs (3–5 days)

## Deliverables
- Fully working VS Code extension that:
  - Guides user through model selection and venv creation on first run.
  - Downloads and verifies model.
  - Runs local Python server (CPU default, optional CUDA).
  - Provides "Predict text from cursor" command integrated into editor.
- README with setup and troubleshooting.
- plan.md (this file) and optional reference mappings to provided sourcemap.
