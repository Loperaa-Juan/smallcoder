# SmallCoder

> Fine-tuning small language models for local, offline code assistance

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HF-smallcoder--dataset-FFD21E?style=flat&logo=huggingface&logoColor=black)
![HuggingFace](https://img.shields.io/badge/HF-qwen2.5--1.5B--code--adapter-FFD21E?style=flat&logo=huggingface&logoColor=black)
![VS Code](https://img.shields.io/badge/VS%20Code-Extension-007ACC?style=flat&logo=visualstudiocode&logoColor=white)
![Status](https://img.shields.io/badge/Status-In%20Progress-orange?style=flat)

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Supported Tasks](#supported-tasks)
- [Datasets](#datasets)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [VS Code Extension](#vs-code-extension)
- [Results](#results)
- [Roadmap](#roadmap)
- [Authors](#authors)

---

## Overview

SmallCoder is a research project that fine-tunes **Qwen2.5-1.5B with LoRA** on code assistance tasks and ships the resulting model inside a **VS Code extension** that runs entirely locally, with no cloud dependency.

---

## How It Works

```
         CodeSearchNet (public dataset)
                      │
                      ▼
             Data Pipeline
          (code_dataset.ipynb)
         1,000 ex / language
            85% / 10% / 5%
                      │
                      ▼
           Unified SmallCoder Dataset
          (Juanxxo/smallcoder-dataset)
          15,250 train / 1,794 val / 897 test
                      │
                      ▼
          Qwen2.5-1.5B-Instruct
           LoRA (4-bit QLoRA)
                      │
                      ▼
              Evaluation
           (BLEU, exact match)
                      │
                      ▼
          Fine-tuned model → VS Code Extension
           (inline predictions, offline)
```

---

## Supported Tasks

| Task | Description | Input Format |
|---|---|---|
| **Code completion** | Predicts the rest of a function from its beginning | `complete <lang>: <partial code>` |
| **Docstring generation** | Generates documentation for a complete function | `generate docstring <lang>: <function>` |

Supported languages: **Python, JavaScript, Java, Go, Ruby, and PHP**.

---

## Datasets

Datasets are built from [CodeSearchNet](https://huggingface.co/datasets/claudios/code_search_net) and published to Hugging Face Hub. Three prefix-cut strategies are used for code completion:

1. First line of the function as prefix.
2. First half of the function as prefix.
3. A reproducible random cut (seed 42).

| Examples per language | Train / Val / Test | HF Hub |
|---|---|---|
| 1,000 | 15,250 / 1,794 / 897 | [`Juanxxo/smallcoder-dataset`](https://huggingface.co/datasets/Juanxxo/smallcoder-dataset) |

Each source function produces up to 4 pairs — three completion cuts and one docstring entry — so the total pair count exceeds the raw function count.

---

## Tech Stack

**ML & Data**
`Python 3.10+` · `PyTorch` · `HuggingFace Transformers` · `PEFT / LoRA` · `TRL (SFTTrainer)` · `BitsAndBytes` · `Datasets` · `Google Colab`

**Extension**
`TypeScript` · `VS Code Extension API` · `Node.js`

---

## Repository Structure

```
SmallCoder/
├── notebooks/
│   ├── code_dataset.ipynb         # Dataset construction and upload pipeline
│   ├── slm_fine_tunning.ipynb     # Qwen2.5-1.5B LoRA fine-tuning on Google Colab
│   └── CodeBLEU_benchmark.ipynb   # CodeBLEU evaluation: base vs. fine-tuned
└── small-coder-extension/         # VS Code extension (in development)
    ├── src/
    │   └── extension.ts         # Extension entry point
    ├── output/                  # Compiled artifacts (TypeScript → JS)
    ├── package.json
    └── tsconfig.json
```

### `notebooks/code_dataset.ipynb`

Google Colab notebook that:

1. Loads CodeSearchNet by language.
2. Generates `(input_code, target_completion)` pairs for completion and docstring tasks.
3. Splits the dataset into train / validation / test (85% / 10% / 5%).
4. Saves a copy to Google Drive and publishes a single unified dataset to Hugging Face Hub.

### `notebooks/slm_fine_tunning.ipynb`

Google Colab notebook that fine-tunes **Qwen2.5-1.5B-Instruct** on the SmallCoder dataset using LoRA with 4-bit quantization (QLoRA). Key details:

- **LoRA config:** r=8, alpha=16, dropout=0.05, all attention and MLP projections targeted
- **Trainable parameters:** 9.23M / 1,552.95M (0.59%)
- **Training:** 3 epochs, batch size 4, sequence length 256, paged AdamW 8-bit optimizer
- **Fine-tuned adapter published to:** [`Juanxxo/qwen2.5-1.5B-code-adapter`](https://huggingface.co/Juanxxo/qwen2.5-1.5B-code-adapter)

### `small-coder-extension/`

VS Code extension written in TypeScript that runs the fine-tuned model **fully locally** — no code ever leaves the machine. See [VS Code Extension](#vs-code-extension) for details.

---

## VS Code Extension

The `small-coder-extension/` directory contains a VS Code extension that brings the fine-tuned SmallCoder model into the editor. The extension host stays lightweight: all inference runs in a local Python server, keeping VS Code responsive.

### Architecture

```
VS Code (TypeScript)                Local Python server
─────────────────────               ────────────────────
extension.ts  ──spawn──►  server/server_launcher.py
predict.ts    ──POST /predict──►    HTTP @ 127.0.0.1 (localhost only)
                          ◄──completion──
```

The extension manages a Python virtual environment, downloads the model, and launches `server_launcher.py` as a background process. The server exposes a minimal HTTP API (`POST /predict`, `GET /health`) bound to `localhost` only, so source code never leaves the machine. When a CUDA GPU is available, the base model is loaded in **4-bit (nf4) with bitsandbytes**; otherwise it falls back to an unquantized CPU load. The fine-tuned [LoRA adapter](https://huggingface.co/Juanxxo/qwen2.5-1.5B-code-adapter) is applied on top via PEFT.

### Commands

| Command | Description |
|---|---|
| `SmallCoder: Setup runtime` | Create the Python venv and prepare the local server |
| `SmallCoder: Predict from cursor` | Send the current editor context to the server and insert the completion |
| `SmallCoder: Force CPU` | Run the runtime on CPU |
| `SmallCoder: Force CUDA` | Run the runtime on an NVIDIA GPU (when available) |
| `SmallCoder: Download model` | Download / prepare the model in extension storage |
| `SmallCoder: Open runtime logs` | Open the SmallCoder output channel |

### Settings

| Setting | Default | Description |
|---|---|---|
| `smallCoder.model` | `smallcoder-tiny` | Model id for the local inference server |
| `smallCoder.envPath` | `""` | Path to the Python virtual environment |
| `smallCoder.device` | `cpu` | Device preference (`cpu` / `cuda`) |
| `smallCoder.maxTokens` | `256` | Max tokens requested per prediction |
| `smallCoder.autoCreateVenv` | `false` | Auto-create the venv during setup |

> **Privacy:** the inference server binds to `127.0.0.1` only and never makes outbound calls with your code — everything runs offline.

---

## Results

### Training

| Model | Training Loss | Params |
|---|---|---|
| Qwen2.5-1.5B (LoRA) | 0.310 | 1.5B (9.23M trainable) |

### CodeBLEU Benchmark

Evaluated on 100 test samples from [`Juanxxo/smallcoder-dataset`](https://huggingface.co/datasets/Juanxxo/smallcoder-dataset) using the [CodeBLEU](https://github.com/microsoft/CodeBLEU) metric (weights 0.25 each). See [`notebooks/CodeBLEU_benchmark.ipynb`](notebooks/CodeBLEU_benchmark.ipynb) for full details.

CodeBLEU combines four sub-metrics, each in **[0, 1]** (higher = closer to the reference):

- **N-gram match** — classic BLEU token n-gram overlap (lexical similarity).
- **Weighted n-gram match** — same, weighting language keywords (`if`, `for`, `return`, …) more heavily.
- **Syntax match** — compares the abstract syntax trees (AST) (structural similarity).
- **Dataflow match** — compares how variables are defined and used (semantic similarity).

The final CodeBLEU is their weighted average.

| Model | CodeBLEU | n-gram | weighted n-gram | syntax | dataflow |
|---|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct (base) | 0.2129 | 0.0163 | 0.0521 | 0.3526 | 0.4307 |
| Qwen2.5-1.5B-code-adapter (fine-tuned) | **0.2253** | **0.0925** | **0.1127** | **0.3944** | 0.3017 |
| Delta | **+0.0124** | +0.0762 | +0.0606 | +0.0418 | −0.1290 |

#### Per-language CodeBLEU

| Language | Samples | Base | Fine-tuned | Δ |
|---|---|---|---|---|
| Python | 38 | 0.2046 | **0.2436** | +0.0390 |
| Java | 18 | 0.1703 | 0.1662 | −0.0041 |
| JavaScript | 13 | 0.2329 | 0.2181 | −0.0148 |
| Ruby | 16 | 0.1740 | **0.2138** | +0.0398 |
| Go | 9 | **0.3135** | 0.3074 | −0.0061 |
| PHP | 6 | **0.3029** | 0.2101 | −0.0928 |

---

## Roadmap

- [x] Dataset pipeline — unified dataset (1,000 ex/language) published to HF Hub
- [x] SLM fine-tuning — Qwen2.5-1.5B + LoRA adapter published to HF Hub
- [x] Evaluation (CodeBLEU) — fine-tuned model scores **0.2253** vs base 0.2129 (+0.0124)
- [x] VS Code extension — model integration

---

## Authors

- [Antonio De León Jiménez](https://github.com/AntJZs)
- [Juan José Lopera](https://github.com/Loperaa-Juan)
