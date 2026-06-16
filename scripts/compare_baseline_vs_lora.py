import pandas as pd
from pathlib import Path


baseline_path = Path("results/baseline_outputs.csv")
lora_path = Path("results/post_finetune_outputs_512_chunked_100.csv")
output_path = Path("results/comparison_baseline_vs_lora_512_chunked_100.csv")

baseline_df = pd.read_csv(baseline_path)
lora_df = pd.read_csv(lora_path)

# Keep only SmolLM2 baseline results
baseline_smollm_df = baseline_df[
    baseline_df["model_name"] == "HuggingFaceTB/SmolLM2-360M"
].copy()

# Keep latest 12 SmolLM2 baseline rows
baseline_smollm_df = baseline_smollm_df.sort_values("run_timestamp").tail(12)

# Create clean comparison columns
baseline_smollm_df["baseline_output_text"] = baseline_smollm_df["generated_text"]
lora_df["lora_output_text"] = lora_df["output_text"]

baseline_smollm_df["baseline_tokens_per_second"] = baseline_smollm_df["tokens_per_second"]
lora_df["lora_tokens_per_second"] = lora_df["tokens_per_second"]

baseline_smollm_df["baseline_gpu_peak_memory_mb"] = baseline_smollm_df["gpu_peak_memory_mb_for_prompt"]
lora_df["lora_gpu_peak_memory_mb"] = lora_df["gpu_peak_memory_mb"]

comparison_df = baseline_smollm_df.merge(
    lora_df,
    on=["prompt_id", "prompt_text", "prompt_type"],
    how="inner",
)

selected_columns = [
    "prompt_id",
    "prompt_type",
    "prompt_text",
    "baseline_output_text",
    "lora_output_text",
    "baseline_tokens_per_second",
    "lora_tokens_per_second",
    "baseline_gpu_peak_memory_mb",
    "lora_gpu_peak_memory_mb",
]

comparison_df = comparison_df[selected_columns]

output_path.parent.mkdir(parents=True, exist_ok=True)
comparison_df.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"Saved comparison to: {output_path}")
print(f"Compared prompts: {len(comparison_df)}")
print(comparison_df[["prompt_id", "prompt_text"]].to_string(index=False))