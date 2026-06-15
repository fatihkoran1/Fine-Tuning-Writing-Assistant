# PRP2 Fine-Tuning Starter

This starter project is for the technical implementation evidence of the PRP2 low-resource writing assistant project.

The first goal is simple:

1. Load one base model.
2. Run fixed baseline prompts.
3. Save the outputs before fine-tuning.
4. Use the same prompts later after LoRA/QLoRA fine-tuning.

## Folder structure

```text
prp2_fine_tuning_starter/
├── configs/
│   ├── baseline_prompts.json
│   └── model_notes.md
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── base/
│   └── fine_tuned/
├── docs
├   ├──DL's/   
│   ├──Project plan and Action plan/ 
│   ├──researchs and evidence/
│   ├──screenshots/
│   ├──short_notes/
│   ├──notebooks/
│       ├── 02_baseline_testing.ipynb
├── results/
├── scripts/
│   └── run_baseline.py
├── src/
├── requirements.txt
├── requirements-qlora-optional.txt
└── README_FIRST_STEP.md
```

## Step 1 - Create and activate the environment

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Step 2 - Check GPU availability

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

This is useful evidence for the model feasibility criterion.

## Step 3 - Run GPT-2 baseline first

```powershell
python scripts/run_baseline.py --model_name openai-community/gpt2
```

The script saves:

```text
results/baseline_outputs.csv
logs/baseline_run_metadata.json
```

## Step 4 - Check the CSV

Open this file:

```text
results/baseline_outputs.csv
```

Important columns:

- `model_name`
- `parameter_count`
- `device`
- `load_seconds`
- `prompt_id`
- `prompt_text`
- `generated_text`
- `continuation_only`
- `generation_seconds`

## Step 5 - Commit this evidence

Suggested Git commit:

```powershell
git add .
git commit -m "test: add GPT-2 baseline generation script"
```

## Later model commands

Only after GPT-2 works:

```powershell
python scripts/run_baseline.py --model_name bigscience/bloom-560m
python scripts/run_baseline.py --model_name HuggingFaceTB/SmolLM2-360M
```

These commands append to the same CSV, so the models can be compared in one file.

## Important limitation

This baseline test does not prove language quality. It only creates pre-fine-tuning evidence.

Quality, memorization, and generalization should be evaluated later by comparing:

- baseline outputs
- post-LoRA outputs
- post-QLoRA outputs if needed
- similarity between generated outputs and training data
