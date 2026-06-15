"""
Baseline model testing script for PRP2 Fine-Tuning project.

Goal:
- Load one causal language model.
- Run the same fixed prompts.
- Save the generated outputs to results/baseline_outputs.csv.
- Save simple timing evidence.

This script does not fine-tune the model.
It only creates baseline outputs before fine-tuning.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


def get_project_root() -> Path:
    """Return the project root based on this file location."""
    return Path(__file__).resolve().parents[1]


def load_prompts(prompts_path: Path) -> list[dict[str, str]]:
    """Load fixed baseline prompts from a JSON file."""
    if not prompts_path.exists():
        raise FileNotFoundError(f"Prompts file was not found: {prompts_path}")

    with prompts_path.open("r", encoding="utf-8") as file:
        prompts = json.load(file)

    if not isinstance(prompts, list):
        raise ValueError("The prompts file must contain a list of prompt objects.")

    required_keys = {"prompt_id", "prompt_type", "prompt_text"}
    for prompt in prompts:
        missing_keys = required_keys - set(prompt.keys())
        if missing_keys:
            raise ValueError(f"Prompt is missing keys: {missing_keys}")

    return prompts


def choose_device() -> str:
    """Use GPU if available, otherwise use CPU."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model_and_tokenizer(model_name: str, device: str) -> tuple[Any, Any, float]:
    """Load tokenizer and model, then return load time in seconds."""
    start_time = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # Some causal language models, especially GPT-2, do not have a pad token.
    # For generation, using the EOS token as padding is a common practical solution.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )

    model.to(device)
    model.eval()

    load_seconds = round(time.perf_counter() - start_time, 3)
    return tokenizer, model, load_seconds


def count_model_parameters(model: Any) -> int:
    """Count the total number of model parameters."""
    return sum(parameter.numel() for parameter in model.parameters())


def get_gpu_memory_mb() -> float | None:
    """Return allocated GPU memory in MB, or None when CUDA is not available."""
    if not torch.cuda.is_available():
        return None

    memory_mb = torch.cuda.memory_allocated() / (1024**2)
    return round(memory_mb, 2)


def generate_text(
    tokenizer: Any,
    model: Any,
    prompt_text: str,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> dict[str, Any]:
    """Generate text for one prompt and return output plus measurements."""
    set_seed(seed)

    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        padding=False,
        truncation=True,
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    start_time = time.perf_counter()

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )

    generation_seconds = round(time.perf_counter() - start_time, 3)

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # This removes the prompt from the beginning when the output starts with the prompt.
    if generated_text.startswith(prompt_text):
        continuation_only = generated_text[len(prompt_text):].strip()
    else:
        continuation_only = generated_text.strip()

    input_tokens = int(inputs["input_ids"].shape[1])
    generated_tokens = int(output_ids.shape[1] - input_tokens)

    return {
        "generated_text": generated_text,
        "continuation_only": continuation_only,
        "input_tokens": input_tokens,
        "generated_tokens": generated_tokens,
        "generation_seconds": generation_seconds,
    }


def save_results(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Save or append baseline results to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    new_results = pd.DataFrame(rows)

    if output_path.exists():
        old_results = pd.read_csv(output_path)
        combined_results = pd.concat([old_results, new_results], ignore_index=True)
        combined_results.to_csv(output_path, index=False, encoding="utf-8")
    else:
        new_results.to_csv(output_path, index=False, encoding="utf-8")


def save_run_metadata(metadata: dict[str, Any], metadata_path: Path) -> None:
    """Save run metadata as JSON evidence."""
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)


def run_baseline(
    model_name: str,
    prompts_path: Path,
    output_path: Path,
    metadata_path: Path,
    max_new_tokens: int = 80,
    temperature: float = 0.7,
    top_p: float = 0.9,
    seed: int = 42,
) -> None:
    """Run baseline generation for one model."""
    run_timestamp = datetime.now().isoformat(timespec="seconds")
    device = choose_device()

    print(f"Loading model: {model_name}")
    print(f"Device: {device}")

    prompts = load_prompts(prompts_path)
    tokenizer, model, load_seconds = load_model_and_tokenizer(model_name, device)
    parameter_count = count_model_parameters(model)

    print(f"Model loaded in {load_seconds} seconds")
    print(f"Parameter count: {parameter_count:,}")

    rows: list[dict[str, Any]] = []

    for prompt in prompts:
        print(f"Generating for prompt: {prompt['prompt_id']}")

        generation = generate_text(
            tokenizer=tokenizer,
            model=model,
            prompt_text=prompt["prompt_text"],
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )

        row = {
            "run_timestamp": run_timestamp,
            "model_name": model_name,
            "parameter_count": parameter_count,
            "device": device,
            "gpu_memory_mb_after_loading": get_gpu_memory_mb(),
            "load_seconds": load_seconds,
            "prompt_id": prompt["prompt_id"],
            "prompt_type": prompt["prompt_type"],
            "prompt_text": prompt["prompt_text"],
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
            **generation,
        }

        rows.append(row)

    save_results(rows, output_path)

    metadata = {
        "run_timestamp": run_timestamp,
        "model_name": model_name,
        "parameter_count": parameter_count,
        "device": device,
        "load_seconds": load_seconds,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "prompts_path": str(prompts_path),
        "output_path": str(output_path),
        "note": "Baseline output before fine-tuning. No training was done in this run.",
    }

    save_run_metadata(metadata, metadata_path)

    print(f"Saved baseline outputs to: {output_path}")
    print(f"Saved run metadata to: {metadata_path}")


def parse_args() -> argparse.Namespace:
    """Read command line arguments."""
    project_root = get_project_root()

    parser = argparse.ArgumentParser(description="Run baseline text generation for one model.")

    parser.add_argument(
        "--model_name",
        type=str,
        default="openai-community/gpt2",
        help="Hugging Face model ID. Default: openai-community/gpt2",
    )

    parser.add_argument(
        "--prompts_path",
        type=Path,
        default=project_root / "configs" / "baseline_prompts.json",
        help="Path to the fixed prompt JSON file.",
    )

    parser.add_argument(
        "--output_path",
        type=Path,
        default=project_root / "results" / "baseline_outputs.csv",
        help="Path where baseline outputs will be saved.",
    )

    parser.add_argument(
        "--metadata_path",
        type=Path,
        default=project_root / "logs" / "baseline_run_metadata.json",
        help="Path where run metadata will be saved.",
    )

    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_baseline(
        model_name=args.model_name,
        prompts_path=args.prompts_path,
        output_path=args.output_path,
        metadata_path=args.metadata_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )
