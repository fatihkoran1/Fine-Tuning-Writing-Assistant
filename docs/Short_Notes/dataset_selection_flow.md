# Dataset Options and Updated Dataset Strategy

## Purpose of this document

This document explains how my dataset strategy developed during PRP2. I started with two options, FLORES+ and AfroLingu-MT, but after investigating FLORES+ more carefully, I found it was not suitable as training data. This led me to Guarani Wikipedia as my final training dataset.

The goal is not only to describe the final choice, but to show the reasoning behind the change.

---

## Project context

My PRP2 project is about fine-tuning a small language model for a low-resource language writing assistant, inspired by Guarani Mbya. I will not use sensitive community data. Instead, I use safe public data to test the core technical question:

> Can a small fine-tuned model adapt to limited language data, or does it mostly memorize?

The dataset needs to be suitable for training, small enough for a low-resource experiment, safe and publicly available, technically usable in Python, splittable into train/validation/test sets, and realistic for a 19-day student project.

---

## Initial dataset options

At the start, I considered two options: FLORES+ and AfroLingu-MT. Guarani Wikipedia was not part of my first options, I found it later after ruling out FLORES+.

---

## Option 1: FLORES+

**Link:** https://huggingface.co/datasets/openlanguagedata/flores_plus  
**Type:** Parallel multilingual / low-resource translation dataset

### Why I first considered it

FLORES+ looked like a good fit because it is already structured, multilingual, and connected to low-resource language work, easy to explain in a portfolio and safe to use without collecting community data. Compared to AfroLingu-MT, it also seemed broader and less complex: AfroLingu-MT required choosing an African language pair and understanding an MT-specific setup, which would add unnecessary scope.

### Why I decided not to use it for training

After checking it more carefully, I realized FLORES+ is a benchmark-style dataset, better suited for evaluation than training. Using it as training data creates data leakage: the model would be trained on material too similar to what is later used for evaluation, making results less trustworthy.

This directly conflicts with my project goal of testing whether the model generalizes or memorizes.

### Current role of FLORES+

Not used as training data. It remains an optional evaluation reference if it fits the final setup.

---

## Option 2: AfroLingu-MT

**Link:** https://huggingface.co/datasets/UBC-NLP/AfroLingu-MT  
**Type:** African low-resource machine translation dataset

### Why I considered it

AfroLingu-MT fits the low-resource theme and is connected to machine translation research, which made it a reasonable comparison option.

### Why I did not choose it

It would require choosing a language pair and shifting the project into an African MT direction, extra complexity for a 19-day project and a move away from the Guarani-inspired context. FLORES+ was the simpler first option for those reasons. AfroLingu-MT could be useful in a project specifically focused on African machine translation, but not here.

---

## What changed after investigating FLORES+

After realizing FLORES+ was not suitable for training, I needed a real text corpus. I searched for a public Guarani dataset and found Guarani Wikipedia.

---

## New training dataset: Guarani Wikipedia

| Field | Value |
|---|---|
| Dataset | `wikimedia/wikipedia` |
| Subset | `20231101.gn` |
| Language | Paraguayan Guarani |
| Rows before cleaning | 5,519 |
| Rows after cleaning | 3,535 |
| Useful columns | `id`, `url`, `title`, `text` |

### Why I chose it

Guarani Wikipedia contains real Guarani text and is better suited for language model fine-tuning than a benchmark-style dataset. The size (after cleaning) falls within the expected low-resource range of 1,000–5,000 examples. It can be loaded with the Hugging Face `datasets` library and cleaned in Python.

### Evidence from my test

I loaded the dataset in Python and inspected sample rows. The dataset loaded successfully and contained real Guarani Wikipedia article text. Cleaning reduced rows from 5,519 to 3,535, filtering out missing, empty, very short, and duplicate texts. The larger-than-expected reduction makes sense because Wikipedia contains many short stubs that do not provide enough language context for fine-tuning.

### Dataset size decision

I will use 3,000 examples from the cleaned 3,535. This is large enough to give the model meaningful language patterns, but controlled enough for the available time and hardware.

---

## Final Dataset Strategy

| Role | Dataset |
|---|---|
| Training | Guarani Wikipedia (`wikimedia/wikipedia`, subset `20231101.gn`) |
| Validation | Held-out split from Guarani Wikipedia |
| Test | Held-out split from Guarani Wikipedia |
| Optional evaluation reference | FLORES+ (not training data) |

---

## Important limitations

- This dataset is **Paraguayan Guarani**, not Guarani Mbya. This is a deliberate and responsible choice: using sensitive Indigenous community data without consent would be an ethical problem. Guarani Mbya is the inspiration and ethical context for this project, not the technical target. The core technical question, whether a small model fine-tuned on limited data can generalize rather than memorize, does not depend on the specific language. Paraguayan Guarani is a suitable and safe stand-in for testing that question.
- Wikipedia text is encyclopedic, not natural writing assistant conversation text.
- I do not speak the language, so I cannot judge linguistic quality.

My evaluation will therefore focus on technical behavior, memorization, generalization, and output comparison, rather than language correctness.