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
│   ├── code_dataset.ipynb       # Dataset construction and upload pipeline
│   └── slm_fine_tunning.ipynb   # Qwen2.5-1.5B LoRA fine-tuning on Google Colab
└── small-coder-extension/       # VS Code extension (in development)
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

VS Code extension written in TypeScript. Currently registers the command `SmallCoder: Predict from the cursor`. The plan is to connect it to the fine-tuned model for inline predictions.

---

## Results

| Model | Training Loss | BLEU (completion) | BLEU (docstring) | Exact Match | Params |
|---|---|---|---|---|---|
| Qwen2.5-1.5B (LoRA) | 0.310 | — | — | — | 1.5B (9.23M trainable) |

---

## Roadmap

- [x] Dataset pipeline — unified dataset (1,000 ex/language) published to HF Hub
- [x] SLM fine-tuning — Qwen2.5-1.5B + LoRA adapter published to HF Hub
- [ ] Evaluation (BLEU with CodeBLEU)
- [ ] VS Code extension — model integration
- [ ] VS Code extension — publish to marketplace

---

## Authors

- [Antonio De León Jiménez](https://github.com/AntJZs)
- [Juan José Lopera](https://github.com/Loperaa-Juan)
