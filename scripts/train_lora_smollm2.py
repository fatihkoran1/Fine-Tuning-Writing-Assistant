import argparse
import json
import time
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def get_peak_gpu_memory_mb():
    if not torch.cuda.is_available():
        return None

    return round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2)


def count_trainable_parameters(model):
    trainable_parameters = 0
    total_parameters = 0

    for parameter in model.parameters():
        total_parameters += parameter.numel()

        if parameter.requires_grad:
            trainable_parameters += parameter.numel()

    trainable_percentage = 100 * trainable_parameters / total_parameters

    return trainable_parameters, total_parameters, trainable_percentage


def tokenize_and_chunk_function(examples, tokenizer, max_seq_length):
    all_input_ids = []
    all_attention_masks = []

    for text in examples["text"]:
        tokenized = tokenizer(
            text,
            truncation=False,
            padding=False,
        )

        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        for start_index in range(0, len(input_ids), max_seq_length):
            end_index = start_index + max_seq_length

            input_chunk = input_ids[start_index:end_index]
            attention_chunk = attention_mask[start_index:end_index]

            if len(input_chunk) < 32:
                continue

            all_input_ids.append(input_chunk)
            all_attention_masks.append(attention_chunk)

    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_masks,
    }


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning smoke test for SmolLM2.")

    parser.add_argument("--model_name", default="HuggingFaceTB/SmolLM2-360M")
    parser.add_argument("--train_file", default="data/processed/train.jsonl")
    parser.add_argument("--validation_file", default="data/processed/validation.jsonl")
    parser.add_argument("--output_dir", default="models/fine_tuned/smollm2_lora_512_smoke")
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--max_steps", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This LoRA test should run on GPU.")

    torch.cuda.reset_peak_memory_stats()

    start_time = time.time()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model: {args.model_name}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Max sequence length: {args.max_seq_length}")
    print(f"Max steps: {args.max_steps}")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        cache_dir="models/base/huggingface_cache",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        cache_dir="models/base/huggingface_cache",
    )

    model.config.use_cache = False

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    trainable_parameters, total_parameters, trainable_percentage = count_trainable_parameters(model)

    print(f"Total parameters: {total_parameters:,}")
    print(f"Trainable parameters: {trainable_parameters:,}")
    print(f"Trainable percentage: {trainable_percentage:.4f}%")

    dataset = load_dataset(
        "json",
        data_files={
            "train": args.train_file,
            "validation": args.validation_file,
        },
    )

    tokenized_dataset = dataset.map(
    lambda examples: tokenize_and_chunk_function(examples, tokenizer, args.max_seq_length),
    batched=True,
    remove_columns=dataset["train"].column_names,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        fp16=True,
        logging_steps=5,
        eval_steps=args.max_steps,
        save_steps=args.max_steps,
        eval_strategy="steps",
        save_strategy="steps",
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
    )

    trainer.train()

    eval_results = trainer.evaluate()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    training_seconds = round(time.time() - start_time, 2)
    peak_gpu_memory_mb = get_peak_gpu_memory_mb()

    metadata = {
        "model_name": args.model_name,
        "output_dir": str(output_dir),
        "max_seq_length": args.max_seq_length,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "training_seconds": training_seconds,
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "trainable_percentage": trainable_percentage,
        "eval_results": eval_results,
    }

    log_path = Path("logs/lora_smoke_tests.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "a", encoding="utf-8") as file:
        file.write(json.dumps(metadata, ensure_ascii=False) + "\n")

    print("Smoke test completed.")
    print(f"Adapter saved to: {output_dir}")
    print(f"Metadata saved to: {log_path}")
    print(f"Training seconds: {training_seconds}")
    print(f"Peak GPU memory MB: {peak_gpu_memory_mb}")
    print(f"Evaluation results: {eval_results}")


if __name__ == "__main__":
    main()