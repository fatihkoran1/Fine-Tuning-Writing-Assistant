"""
Baseline model testing script for the PRP2 Fine-Tuning Writing Assistant project.

Goal:
- Load one causal language model.
- Run the same fixed prompts for every model.
- Save generated baseline outputs to results/baseline_outputs.csv.
- Save run metadata to logs/baseline_run_metadata.jsonl.

Important:
This script does not fine-tune the model.
It only creates baseline evidence before LoRA or QLoRA training.
"""

from __future__ import annotations

import argparse
import gc
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
        raise FileNotFoundError(
            f"Prompts file was not found: {prompts_path}\n"
            "Create configs/baseline_prompts.json before running baseline testing."
        )

    with prompts_path.open("r", encoding="utf-8") as file:
        prompts = json.load(file)

    if not isinstance(prompts, list):
        raise ValueError("The prompts file must contain a list of prompt objects.")

    required_keys = {"prompt_id", "prompt_type", "prompt_text"}

    for index, prompt in enumerate(prompts):
        if not isinstance(prompt, dict):
            raise ValueError(f"Prompt at index {index} must be a JSON object.")

        missing_keys = required_keys - set(prompt.keys())
        if missing_keys:
            raise ValueError(f"Prompt at index {index} is missing keys: {missing_keys}")

        if not prompt["prompt_text"].strip():
            raise ValueError(f"Prompt at index {index} has an empty prompt_text value.")

    return prompts


def choose_device(force_cpu: bool = False) -> str:
    """Use GPU if available, unless force_cpu is enabled."""
    if force_cpu:
        return "cpu"

    return "cuda" if torch.cuda.is_available() else "cpu"


def get_torch_dtype(device: str) -> torch.dtype:
    """Use float16 on GPU and float32 on CPU."""
    if device == "cuda":
        return torch.float16

    return torch.float32


def load_model_and_tokenizer(
    model_name: str,
    device: str,
    cache_dir: Path | None,
) -> tuple[Any, Any, float]:
    """Load tokenizer and model, then return load time in seconds."""
    start_time = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        cache_dir=cache_dir,
    )

    # Some causal language models, especially GPT-2, do not have a pad token.
    # For generation, using the EOS token as padding is a common practical solution.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_dtype = get_torch_dtype(device)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=model_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
            cache_dir=cache_dir,
        )
    except TypeError:
        # Older Transformers versions may not support the dtype argument yet.
        # In that case, use torch_dtype as a fallback.
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=model_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
            cache_dir=cache_dir,
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


def get_gpu_peak_memory_mb() -> float | None:
    """Return peak allocated GPU memory in MB, or None when CUDA is not available."""
    if not torch.cuda.is_available():
        return None

    memory_mb = torch.cuda.max_memory_allocated() / (1024**2)
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
    input_token_count = int(inputs["input_ids"].shape[1])

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    start_time = time.perf_counter()

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generation_seconds = round(time.perf_counter() - start_time, 3)

    # Decode the full output for reading and portfolio evidence.
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Decode only the newly generated token IDs.
    # This is safer than string slicing because tokenizers can normalize whitespace
    # or characters differently during encode/decode.
    new_token_ids = output_ids[0][input_token_count:]
    continuation_only = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
    generated_token_count = int(new_token_ids.shape[0])

    tokens_per_second = None
    if generation_seconds > 0:
        tokens_per_second = round(generated_token_count / generation_seconds, 3)

    return {
        "generated_text": generated_text,
        "continuation_only": continuation_only,
        "input_tokens": input_token_count,
        "generated_tokens": generated_token_count,
        "generation_seconds": generation_seconds,
        "tokens_per_second": tokens_per_second,
        "gpu_memory_mb_after_generation": get_gpu_memory_mb(),
        "gpu_peak_memory_mb_for_prompt": get_gpu_peak_memory_mb(),
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
    """Append run metadata as JSON Lines evidence."""
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    with metadata_path.open("a", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False)
        file.write("\n")


def clear_memory() -> None:
    """Release unused memory after the script finishes."""
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_baseline(
    model_name: str,
    prompts_path: Path,
    output_path: Path,
    metadata_path: Path,
    cache_dir: Path | None,
    max_new_tokens: int = 80,
    temperature: float = 0.7,
    top_p: float = 0.9,
    seed: int = 42,
    force_cpu: bool = False,
) -> None:
    """Run baseline generation for one model."""
    run_timestamp = datetime.now().isoformat(timespec="seconds")
    device = choose_device(force_cpu=force_cpu)

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {model_name}")
    print(f"Device: {device}")

    if device == "cuda":
        print(f"CUDA GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA GPU: not available in this Python environment")

    print(f"Cache directory: {cache_dir if cache_dir else 'Default Hugging Face cache'}")

    prompts = load_prompts(prompts_path)
    tokenizer, model, load_seconds = load_model_and_tokenizer(
        model_name=model_name,
        device=device,
        cache_dir=cache_dir,
    )
    parameter_count = count_model_parameters(model)
    gpu_memory_after_loading = get_gpu_memory_mb()

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
            "gpu_memory_mb_after_loading": gpu_memory_after_loading,
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
        "gpu_memory_mb_after_loading": gpu_memory_after_loading,
        "load_seconds": load_seconds,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "prompts_path": str(prompts_path),
        "output_path": str(output_path),
        "cache_dir": str(cache_dir) if cache_dir else None,
        "number_of_prompts": len(prompts),
        "note": "Baseline output before fine-tuning. No training was done in this run.",
    }

    save_run_metadata(metadata, metadata_path)

    print(f"Saved baseline outputs to: {output_path}")
    print(f"Appended run metadata to: {metadata_path}")

    clear_memory()


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
        default=project_root / "logs" / "baseline_run_metadata.jsonl",
        help="Path where run metadata will be appended as JSON Lines.",
    )

    parser.add_argument(
        "--cache_dir",
        type=Path,
        default=project_root / "models" / "base" / "huggingface_cache",
        help="Directory where Hugging Face model files will be cached.",
    )

    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--force_cpu",
        action="store_true",
        help="Force the script to run on CPU even if CUDA is available.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_baseline(
        model_name=args.model_name,
        prompts_path=args.prompts_path,
        output_path=args.output_path,
        metadata_path=args.metadata_path,
        cache_dir=args.cache_dir,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        force_cpu=args.force_cpu,
    )
