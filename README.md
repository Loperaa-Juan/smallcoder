# SmallCoder

SmallCoder es un proyecto de investigación que compara el fine-tuning de dos familias de modelos de lenguaje pequeños para tareas de asistencia de código: **T5-large** y un modelo **SLM con QLoRA**. El objetivo final es integrar el modelo resultante en una extensión de VS Code que ofrezca predicciones de código directamente en el editor, de forma local y sin depender de servicios en la nube.

---

## Objetivos

- Construir un pipeline reproducible de datos a partir del dataset público [CodeSearchNet](https://huggingface.co/datasets/claudios/code_search_net).
- Entrenar y comparar T5-large (fine-tuning estándar) vs. un SLM con QLoRA sobre las mismas tareas.
- Exponer el modelo ganador a través de una extensión de VS Code.

---

## Tareas soportadas

| Tarea | Descripción | Formato de entrada |
|---|---|---|
| **Completado de código** | Predice el resto de una función a partir de su inicio | `complete <lang>: <código parcial>` |
| **Generación de docstring** | Genera la documentación de una función completa | `generate docstring <lang>: <función>` |

Los lenguajes cubiertos son: **Python, JavaScript, Java, Go, Ruby y PHP**.

---

## Datasets

Los datasets se construyen a partir de CodeSearchNet y se publican en Hugging Face Hub con tres estrategias de corte para el completado de código:

1. Primera línea de la función como prefijo.
2. Primera mitad de la función como prefijo.
3. Un corte aleatorio reproducible (seed 42).

| Modelo objetivo | Ejemplos por lenguaje | Split train / val / test | HF Hub |
|---|---|---|---|
| T5-large | 1 500 | 28 657 / 3 371 / 1 685 | [`Juanxxo/smallcoder-t5-dataset`](https://huggingface.co/datasets/Juanxxo/smallcoder-t5-dataset) |
| SLM (QLoRA) | 3 000 | 57 272 / 6 737 / 3 368 | [`Juanxxo/smallcoder-slm-dataset`](https://huggingface.co/datasets/Juanxxo/smallcoder-slm-dataset) |

El dataset SLM es más grande porque la cuantización de QLoRA permite entrenar con mayor volumen sin problemas de memoria.

---

## Estructura del repositorio

```
T5-fine-tunning/
├── notebooks/
│   └── code_dataset.ipynb      # Pipeline de construcción y subida del dataset
└── small-coder-extension/      # Extensión de VS Code (en desarrollo)
    ├── src/
    │   └── extension.ts        # Punto de entrada de la extensión
    ├── output/                 # Artefactos compilados (TypeScript → JS)
    ├── package.json
    └── tsconfig.json
```

### `notebooks/code_dataset.ipynb`

Notebook de Google Colab que:

1. Carga el dataset base de CodeSearchNet por lenguaje.
2. Genera pares `(input_code, target_completion)` para completado y docstrings.
3. Parte el dataset en train / validation / test (85 % / 10 % / 5 %).
4. Guarda una copia en Google Drive y publica en Hugging Face Hub.

### `small-coder-extension/`

Extensión de VS Code escrita en TypeScript. Actualmente registra el comando `SmallCoder: Predict from the cursor` (implementación pendiente). El plan es conectarla al modelo fine-tuneado para ofrecer predicciones inline.

---

## Autores

- [Antonio De León Jiménez](https://github.com/AntJZs)
- [Juan José Lopera](https://github.com/Loperaa-Juan)