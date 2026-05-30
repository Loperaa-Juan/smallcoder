# SmallCoder

> Fine-tuning small language models for local, offline code assistance

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HF-smallcoder--t5--dataset-FFD21E?style=flat&logo=huggingface&logoColor=black)
![HuggingFace](https://img.shields.io/badge/HF-smallcoder--slm--dataset-FFD21E?style=flat&logo=huggingface&logoColor=black)
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

SmallCoder is a research project that fine-tunes and compares two small language model families — **T5-large** and an **SLM with QLoRA** — on code assistance tasks. Both models are trained on the same data and evaluated under the same conditions to determine which architecture offers the best quality-to-size trade-off. The winning model is then shipped inside a **VS Code extension** that runs entirely locally, with no cloud dependency.

---

## How It Works

```
         CodeSearchNet (public dataset)
                      │
                      ▼
             Data Pipeline
          (code_dataset.ipynb)
            85% / 10% / 5%
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
     T5-large                SLM + QLoRA
  standard fine-tuning    quantized fine-tuning
   1,500 ex / language     3,000 ex / language
          │                       │
          └───────────┬───────────┘
                      ▼
           Evaluation & Comparison
            (BLEU, exact match)
                      │
                      ▼
          Best model → VS Code Extension
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

| Target model | Examples per language | Train / val / test | HF Hub |
|---|---|---|---|
| T5-large | 1,500 | 28,657 / 3,371 / 1,685 | [`Juanxxo/smallcoder-t5-dataset`](https://huggingface.co/datasets/Juanxxo/smallcoder-t5-dataset) |
| SLM (QLoRA) | 3,000 | 57,272 / 6,737 / 3,368 | [`Juanxxo/smallcoder-slm-dataset`](https://huggingface.co/datasets/Juanxxo/smallcoder-slm-dataset) |

The SLM dataset is larger because QLoRA quantization allows training at higher data volumes without running into memory constraints.

---

## Tech Stack

**ML & Data**
`Python 3.10+` · `PyTorch` · `HuggingFace Transformers` · `PEFT / QLoRA` · `Datasets` · `Google Colab`

**Extension**
`TypeScript` · `VS Code Extension API` · `Node.js`

---

## Repository Structure

```
SmallCoder/
├── notebooks/
│   └── code_dataset.ipynb      # Dataset construction and upload pipeline
└── small-coder-extension/      # VS Code extension (in development)
    ├── src/
    │   └── extension.ts        # Extension entry point
    ├── output/                 # Compiled artifacts (TypeScript → JS)
    ├── package.json
    └── tsconfig.json
```

### `notebooks/code_dataset.ipynb`

Google Colab notebook that:

1. Loads CodeSearchNet by language.
2. Generates `(input_code, target_completion)` pairs for completion and docstring tasks.
3. Splits the dataset into train / validation / test (85% / 10% / 5%).
4. Saves a copy to Google Drive and publishes both datasets to Hugging Face Hub.

### `small-coder-extension/`

VS Code extension written in TypeScript. Currently registers the command `SmallCoder: Predict from the cursor`. The plan is to connect it to the fine-tuned model for inline predictions.

---

## Results

> Training in progress — results will be updated once both models complete fine-tuning.

| Model | BLEU (completion) | BLEU (docstring) | Exact Match | Params |
|---|---|---|---|---|
| T5-large | — | — | — | 770M |
| SLM (QLoRA) | — | — | — | TBD |

---

## Roadmap

- [x] Dataset pipeline — T5 and SLM variants published to HF Hub
- [ ] T5-large fine-tuning
- [ ] SLM fine-tuning (QLoRA)
- [ ] Evaluation and model comparison
- [ ] VS Code extension — model integration
- [ ] VS Code extension — publish to marketplace

---

## Authors

- [Antonio De León Jiménez](https://github.com/AntJZs)
- [Juan José Lopera](https://github.com/Loperaa-Juan)
