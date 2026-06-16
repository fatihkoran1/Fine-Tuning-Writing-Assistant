import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def load_prompts(prompt_path):
    with open(prompt_path, "r", encoding="utf-8") as file:
        return json.load(file)


def generate_text(model, tokenizer, prompt_text, device, max_new_tokens, temperature, top_p):
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    input_token_count = inputs["input_ids"].shape[1]

    start_time = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )

    generation_seconds = time.time() - start_time

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    output_token_count = outputs.shape[1]
    new_token_count = output_token_count - input_token_count

    tokens_per_second = 0
    if generation_seconds > 0:
        tokens_per_second = new_token_count / generation_seconds

    return {
        "output_text": output_text,
        "input_token_count": input_token_count,
        "output_token_count": output_token_count,
        "new_token_count": new_token_count,
        "generation_seconds": round(generation_seconds, 3),
        "tokens_per_second": round(tokens_per_second, 3),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base_model_name",
        type=str,
        default="bigscience/bloom-560m",
    )

    parser.add_argument(
        "--adapter_path",
        type=str,
        default="models/fine_tuned/bloom_560m_lora_512_chunked_wide_500",
    )

    parser.add_argument(
        "--prompt_path",
        type=str,
        default="configs/baseline_prompts.json",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="results/post_finetune_outputs_bloom_wide_512_chunked_500.csv",
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=80,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
    )

    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
    )

    args = parser.parse_args()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Base model: {args.base_model_name}")
    print(f"LoRA adapter: {args.adapter_path}")
    print(f"Device: {device}")

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if device == "cuda":
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model_name,
            dtype=torch.float16,
            device_map={"": 0},
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(args.base_model_name)

    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.eval()

    parameter_count = count_parameters(model)

    prompts = load_prompts(args.prompt_path)

    rows = []
    run_timestamp = datetime.now().isoformat(timespec="seconds")

    for prompt in prompts:
        prompt_id = prompt["prompt_id"]
        prompt_type = prompt["prompt_type"]
        prompt_text = prompt["prompt_text"]

        print(f"Generating: {prompt_id} - {prompt_text}")

        result = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        gpu_peak_memory_mb = None

        if device == "cuda":
            gpu_peak_memory_mb = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2)

        row = {
            "run_timestamp": run_timestamp,
            "base_model_name": args.base_model_name,
            "adapter_path": args.adapter_path,
            "prompt_id": prompt_id,
            "prompt_type": prompt_type,
            "prompt_text": prompt_text,
            "output_text": result["output_text"],
            "device": device,
            "parameter_count": parameter_count,
            "input_token_count": result["input_token_count"],
            "output_token_count": result["output_token_count"],
            "new_token_count": result["new_token_count"],
            "generation_seconds": result["generation_seconds"],
            "tokens_per_second": result["tokens_per_second"],
            "gpu_peak_memory_mb": gpu_peak_memory_mb,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        }

        rows.append(row)

    fieldnames = [
        "run_timestamp",
        "base_model_name",
        "adapter_path",
        "prompt_id",
        "prompt_type",
        "prompt_text",
        "output_text",
        "device",
        "parameter_count",
        "input_token_count",
        "output_token_count",
        "new_token_count",
        "generation_seconds",
        "tokens_per_second",
        "gpu_peak_memory_mb",
        "max_new_tokens",
        "temperature",
        "top_p",
    ]

    with open(output_path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved BLOOM LoRA generation outputs to: {output_path}")


if __name__ == "__main__":
    main()