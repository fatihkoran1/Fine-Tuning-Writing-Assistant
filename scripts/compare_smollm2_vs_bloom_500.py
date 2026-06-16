import pandas as pd
from pathlib import Path


smollm2_path = Path("results/post_finetune_outputs_512_chunked_500.csv")
bloom_path = Path("results/post_finetune_outputs_bloom_wide_512_chunked_500.csv")
output_path = Path("results/comparison_smollm2_500_vs_bloom_wide_500.csv")

smollm2_df = pd.read_csv(smollm2_path)
bloom_df = pd.read_csv(bloom_path)

smollm2_df = smollm2_df[
    ["prompt_id", "prompt_type", "prompt_text", "output_text", "tokens_per_second", "gpu_peak_memory_mb"]
].copy()

smollm2_df = smollm2_df.rename(
    columns={
        "output_text": "smollm2_500_output_text",
        "tokens_per_second": "smollm2_tokens_per_second",
        "gpu_peak_memory_mb": "smollm2_generation_gpu_peak_memory_mb",
    }
)

bloom_df = bloom_df[
    ["prompt_id", "prompt_type", "prompt_text", "output_text", "tokens_per_second", "gpu_peak_memory_mb"]
].copy()

bloom_df = bloom_df.rename(
    columns={
        "output_text": "bloom_wide_500_output_text",
        "tokens_per_second": "bloom_tokens_per_second",
        "gpu_peak_memory_mb": "bloom_generation_gpu_peak_memory_mb",
    }
)

comparison_df = smollm2_df.merge(
    bloom_df,
    on=["prompt_id", "prompt_type", "prompt_text"],
    how="inner",
)

comparison_df["prompt_number"] = comparison_df["prompt_id"].str.replace("p", "").astype(int)
comparison_df = comparison_df.sort_values("prompt_number")
comparison_df = comparison_df.drop(columns=["prompt_number"])

output_path.parent.mkdir(parents=True, exist_ok=True)
comparison_df.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"Saved comparison to: {output_path}")
print(f"Compared prompts: {len(comparison_df)}")
print(comparison_df[["prompt_id", "prompt_text"]].to_string(index=False))