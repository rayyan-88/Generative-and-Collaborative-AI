# Amazon Review Summarization with FLAN-T5 + LoRA

A Colab/Jupyter notebook project that compares **baseline generation**, **prompt engineering**, and **LoRA fine-tuning** for abstractive summarization of Amazon product reviews.

The goal is to turn a long Amazon review into a short, headline-style summary.

**Example**

- **Review:** "I bought this for my daughter and she loves it! The color is vibrant and it fits perfectly."
- **Summary:** "Great gift, perfect fit!"

---

## Project Overview

This project implements three approaches:

1. **Baseline model** — `google/flan-t5-small` without fine-tuning.
2. **Prompt engineering** — zero-shot, refined zero-shot, and few-shot prompting.
3. **Fine-tuned model** — LoRA / PEFT adapter trained on Amazon review headlines.

The notebook evaluates each approach using ROUGE and BERTScore.

---

## Notebook Structure

The code is organized as sequential notebook cells. Run them in order.

| Cell | Purpose | Key Output |
|------|---------|------------|
| 1 | Environment setup | Installs dependencies, checks GPU |
| 2 | Configuration | `CFG` dataclass, seed, directories |
| 3 | Data loading | Amazon Polarity 10K subset, 80/10/10 split |
| 4 | Preprocessing | Tokenized inputs and labels |
| 5 | Baseline model | Loads FLAN-T5-small |
| 6 | Baseline generation | `baseline_generations.csv` |
| 8 | Baseline evaluation | `baseline_metrics.csv` |
| 9 | Prompt engineering | `prompt_engineering_comparison.csv` |
| 10 | LoRA setup | PEFT model and Trainer |
| 11 | Fine-tuning | Saved LoRA adapter |
| 12 | Final evaluation | `final_model_comparison.csv` |
| 13 | Inference demo | Generates a custom headline |

> Note: Cell 7 is not present in the notebook. The numbering skips from 6 to 8.

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Base model | `google/flan-t5-small` |
| Fine-tuning | PEFT / LoRA |
| Framework | Hugging Face Transformers |
| Dataset | `amazon_polarity` |
| Evaluation | ROUGE, BERTScore |
| Hardware | NVIDIA T4 GPU (recommended) |
| Environment | Google Colab / Jupyter |

---

## Dataset

**Amazon Polarity** from Hugging Face Datasets.

- Source: `amazon_polarity`
- Subset used: 10,000 training examples
- Split: 80% train / 10% validation / 10% test
- Input column: `content` → renamed to `text`
- Target column: `title` → renamed to `summary`

The dataset is mapped for summarization:

```python
dataset = raw_data.map(lambda x: {
    "text": x["content"],
    "summary": x["title"]
}, remove_columns=["content", "title", "label"])
