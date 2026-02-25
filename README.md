# AI-generated Text (AIGT) Detection

This repository contains functions to use a fine tune of `Qwen/Qwen2.5-3B-Instruct` to estimate the ratio of AI assistance within a text (as provided by superset). It includes:

- Token-weighted inference.
- Article-level aggregation.
- API (detect_batch).
- CLI for batch scoring.
- Training data and scripts.

# How the text is processed

## Windowing

- Texts are split into contiguous windows of at most `500` tokens.
- Boundaries are snapped to sentence/paragraphs ends whenever is possible.
- No overlaps.
- No normalization (text from wasabi is used as input, without any preprocessing).

## Aggregation

- Each window provides an `AI/Human` label.
- Article-level AI ratio is computed as `total tokens in ai windows/total tokens in all windows`.
- We omit the last window for article-level aggregation if its length is less than `75` tokens (only if the article has 2 or more windows).
- Taxonomy labels are derived from the AI ratio.

# Languages

Each language has its own LoRA adapter and its own classification head. All languages share the same Qwen backbone. We can implement more languages by uploading safetensors to the hugging face repository and modifying properly the arguments of the `detect_batch` function accordingly.

For a language code not listed in the function dictionary, the detector will default to using the English language adapter. The current list of languages supported:

- English (`en`)
- Spanish; Castilian (`es`)
- Italian (`it`)
- French (`fr`)
- German (`de`)
- Japanese (`ja`)

# Taxonomy

The 3-class taxonomy (`prediction_short`) is defined by the rules:

- `AI` if `fraction_ai` > `0.8`
- `Mixed` if `0.3` < `fraction_ai` <= `0.8`
- `Human` if `fraction_ai` <= `0.3`

The 6-class taxonomy (`prediction_long`) is defined by the rules:

- `Fully AI-generated` if `fraction_ai` = `1`
- `Primarily AI; some human detected` if `0.8` < `fraction_ai` < `1`
- `Mix of AI and human` if `0.3` < `fraction_ai` <= `0.8`
- `Primarily human; some AI detected` if `0.1` < `fraction_ai` <= `0.3`
- `Primarily human; small AI detected` if `0` < `fraction_ai` <= `0.1`
- `Fully human-written` if `fraction_ai` = `0`

# Installation

```bash
pip install git+https://github.com/deepsee-code/ai-gen-text-classifier
```

# Example

The following code

```python
from aigt import detect_batch
import torch

hf_token = "YOUR TOKEN"

texts = [
    '''Hello world!''',
    '''In a quiet corner of the city, there was a small bookstore that smelled like old paper and rain. People came there not only to buy books, but to escape the noise of the world.
    The owner, an old man with kind eyes, always seemed to know exactly which story someone needed. And somehow, every visitor left a little lighter than when they arrived.''',
    '''El Dr. Capy Cosmos, un capybara diferente a cualquier otro, asombró a la comunidad científica con su innovadora investigación en astrofísica.
    Con su agudo sentido de la observación y su capacidad sin igual para interpretar datos cósmicos, descubrió nuevas pistas sobre los misterios de los agujeros negros y el origen del universo.
    Mientras observaba a través de telescopios con sus grandes ojos redondos, sus colegas investigadores solían comentar que parecía como si las estrellas mismas le susurraran sus secretos directamente a él.
    El Dr. Cosmos no solo se convirtió en un faro de inspiración para los científicos en formación, sino que también demostró que la inteligencia y la innovación pueden encontrarse en las criaturas más inesperadas.''',
    '''Dans une petite ville au bord de la mer, il y avait un café silencieux où les gens venaient pour rêver. Chaque matin, le soleil entrait doucement par les fenêtres, et l’odeur du pain chaud remplissait l’air.
    Une vieille femme écrivait des lettres qu’elle n’envoyait jamais, tandis qu’un jeune garçon dessinait des bateaux sur une nappe en papier. Personne ne parlait beaucoup, mais tout le monde semblait comprendre que certains moments n’ont pas besoin de mots.'''
    ]

res = detect_batch(
    texts,
    prediction_ids = ["doc1", "doc2", "doc3", "doc4"], # Important: IDs must be unique
    lang = ["en", "en", "es", "fr"],             # language_code as in superset table; not supported languages or unknown language codes will be predicted using the english model.

    # HF / LoRA config
    repo_id = "DeepSee-io/qwen_adapters_aigt", # HF repo storing the safetensors
    subdir_by_lang = {
        "en": "en/best",
        "es": "es/best",
        "fr": "fr/best",
        "de": "de/best",
        "it": "it/best",
        "ja": "ja/best",
    },
    hf_token = hf_token, # token to access the repo storing the safetensors

    max_len = 500,
    batch_size = 16,
    progress = False,
    return_text = False,
)

res
```

will produce

```
[{'prediction_id': 'doc1',
  'lang': 'en',
  'fraction_ai': 0.0,
  'prediction_short': 'Human',
  'prediction_long': 'Fully human-written',
  'n_windows': 1,
  'n_ai_segments': 0,
  'n_human_segments': 1,
  'n_tokens': 3},
 {'prediction_id': 'doc2',
  'lang': 'en',
  'fraction_ai': 1.0,
  'prediction_short': 'AI',
  'prediction_long': 'Fully AI-generated',
  'n_windows': 1,
  'n_ai_segments': 1,
  'n_human_segments': 0,
  'n_tokens': 74},
 {'prediction_id': 'doc3',
  'lang': 'es',
  'fraction_ai': 1.0,
  'prediction_short': 'AI',
  'prediction_long': 'Fully AI-generated',
  'n_windows': 2,
  'n_ai_segments': 2,
  'n_human_segments': 0,
  'n_tokens': 188},
 {'prediction_id': 'doc4',
  'lang': 'fr',
  'fraction_ai': 1.0,
  'prediction_short': 'AI',
  'prediction_long': 'Fully AI-generated',
  'n_windows': 1,
  'n_ai_segments': 1,
  'n_human_segments': 0,
  'n_tokens': 130}]
```
# CLI Usage

```bash
python scripts/predict.py \
  --input input.parquet \
  --text-col text \
  --id-col prediction_id \
  --lang-col lang \
  --repo-id DeepSee-io/qwen_adapters_aigt \
  --token=HF_TOKEN
  --output-prefix demo_run
```

Outputs:
- `demo_run_articles.json`
- `demo_run_windows.json` (optional)

# Training

We used the latest version of Pangram Labs Detector (https://www.pangram.com/blog/introducing-ai-assistance-detection) to get window-level data. When getting labels/scores for English data, only article-level predictions were available because window-level analysis was not supported at that time.

To fine tune the LLM using QLoRA, use:

```bash
python scripts/train_windows.py \
  --data-csv data/qlora_training_data_windows_FR.csv \
  --text-col window_text \
  --target-col ai_assistance_score \
  --group-col target_url \
  --output-dir runs/qlora_fr \
  --val-ratio 0.0
```
Keep in mind that Pangram Detector has a non-zero error rate, so labels are not perfect. Hence, we use all labeled data for training and use different data sets to get FPR and Recall estimates (pre-ChatGPT articles for FPR and Synthetic Mirrors of such articles for Recall).

# Adding more languages

To add a new language we upload LoRA artifacts to a huggingface repository (safetensors exceed the size limit for a file on a GitHub repo).

```pgsql
XX/best/          # XX is the language code to be added
├── lora_adapter/
│   ├── adapter_model.safetensors
│   └── adapter_config.json
├── head.pt
└── train_config.json
```
then we register it at call time:

```python
subdir_by_lang["XX"] = "XX/best"
```


# GUI

A GUI to test the tool on individual texts: https://huggingface.co/spaces/Trinotrotolueno/AIGT_Detector

# Notes

- `<1%` false-positive rate at the article-level (pre-ChatGPT texts).
- More than `90%` of recall for modern LLMs (`chatGPT-5.2`, `Claude 3.5`, `GPT-4o`, etc.).
- Optimized for GPU inference with 4-bit QLoRA to reduce VRAM usage. The function is able to run on lower-end GPUs.
- Languages are processed sequentially to avoid out-of-memory problems.
- Only works for natural language. Not suited for AI-generated code or AI-assisted math solving.
- Model trained on article data, but still able to detect AI artifacts with high precision through boilerplate text or navigation text (commonly observed in homepages).
- The output (`fraction_ai`) shouldn't be interpreted as a probability; instead, it is the ratio of AI-assistance within the whole text.

# Limitations

- Non-zero error rates.
- Best used as a statistical signal at the domain-level.
- Short texts (less than `75` words or `~100` tokens) are inherently noisy.

# To implement

- Training code and training data for English language.
- Code to compute estimates for `FPR` (false-positive rate) and `recall` at the article-level, using pristine articles and synthetic mirrors.
- An active learning pipeline where we use false positives and synthetic mirrors to retrain the models, in hopes of minimizing the false-positive rates while mantaining a small training dataset. 
