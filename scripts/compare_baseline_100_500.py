import pandas as pd
from pathlib import Path


baseline_path = Path("results/baseline_outputs.csv")
lora_100_path = Path("results/post_finetune_outputs_512_chunked_100.csv")
lora_500_path = Path("results/post_finetune_outputs_512_chunked_500.csv")

output_path = Path("results/comparison_baseline_vs_lora_100_vs_500.csv")

baseline_df = pd.read_csv(baseline_path)
lora_100_df = pd.read_csv(lora_100_path)
lora_500_df = pd.read_csv(lora_500_path)

# Keep only SmolLM2 baseline
baseline_smollm_df = baseline_df[
    baseline_df["model_name"] == "HuggingFaceTB/SmolLM2-360M"
].copy()

# Keep latest 12 baseline rows
baseline_smollm_df = baseline_smollm_df.sort_values("run_timestamp").tail(12)

# Prepare clean columns
baseline_smollm_df = baseline_smollm_df[
    ["prompt_id", "prompt_type", "prompt_text", "generated_text", "tokens_per_second"]
].copy()

baseline_smollm_df = baseline_smollm_df.rename(
    columns={
        "generated_text": "baseline_output_text",
        "tokens_per_second": "baseline_tokens_per_second",
    }
)

lora_100_df = lora_100_df[
    ["prompt_id", "prompt_type", "prompt_text", "output_text", "tokens_per_second"]
].copy()

lora_100_df = lora_100_df.rename(
    columns={
        "output_text": "lora_100_output_text",
        "tokens_per_second": "lora_100_tokens_per_second",
    }
)

lora_500_df = lora_500_df[
    ["prompt_id", "prompt_type", "prompt_text", "output_text", "tokens_per_second"]
].copy()

lora_500_df = lora_500_df.rename(
    columns={
        "output_text": "lora_500_output_text",
        "tokens_per_second": "lora_500_tokens_per_second",
    }
)

comparison_df = baseline_smollm_df.merge(
    lora_100_df,
    on=["prompt_id", "prompt_type", "prompt_text"],
    how="inner",
)

comparison_df = comparison_df.merge(
    lora_500_df,
    on=["prompt_id", "prompt_type", "prompt_text"],
    how="inner",
)

# Optional: sort by prompt number p01, p02, ...
comparison_df["prompt_number"] = comparison_df["prompt_id"].str.replace("p", "").astype(int)
comparison_df = comparison_df.sort_values("prompt_number")
comparison_df = comparison_df.drop(columns=["prompt_number"])

output_path.parent.mkdir(parents=True, exist_ok=True)
comparison_df.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"Saved comparison to: {output_path}")
print(f"Compared prompts: {len(comparison_df)}")
print(comparison_df[["prompt_id", "prompt_text"]].to_string(index=False))