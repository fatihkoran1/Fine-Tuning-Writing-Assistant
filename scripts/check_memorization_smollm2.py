import json
import re
from pathlib import Path

import pandas as pd


TRAIN_PATH = Path("data/processed/train.jsonl")
OUTPUT_PATH = Path("results/generalization_outputs_smollm2_500.csv")
REPORT_PATH = Path("results/memorization_check_generalization_smollm2_500.csv")


def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_word_ngrams(text, n=12):
    words = normalize_text(text).split()
    ngrams = []

    for index in range(0, len(words) - n + 1):
        ngram = " ".join(words[index:index + n])
        ngrams.append(ngram)

    return ngrams


def load_training_text():
    parts = []

    with open(TRAIN_PATH, "r", encoding="utf-8") as file:
        for line in file:
            item = json.loads(line)
            text = item.get("text", "")
            parts.append(normalize_text(text))

    return " ".join(parts)


def main():
    print("Loading training data...")
    training_text = load_training_text()

    print("Loading generated outputs...")
    output_df = pd.read_csv(OUTPUT_PATH)

    rows = []

    for _, row in output_df.iterrows():
        prompt_id = row["prompt_id"]
        prompt_text = row["prompt_text"]
        output_text = row["output_text"]

        ngrams = get_word_ngrams(output_text, n=12)

        exact_matches = []

        for ngram in ngrams:
            if ngram in training_text:
                exact_matches.append(ngram)

        rows.append(
            {
                "prompt_id": prompt_id,
                "prompt_text": prompt_text,
                "output_text": output_text,
                "number_of_12_word_exact_matches": len(exact_matches),
                "example_exact_match": exact_matches[0] if exact_matches else "",
                "memorization_risk": "possible" if exact_matches else "not_found",
            }
        )

    report_df = pd.DataFrame(rows)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(REPORT_PATH, index=False, encoding="utf-8-sig")

    print(f"Saved memorization report to: {REPORT_PATH}")
    print(report_df[["prompt_id", "number_of_12_word_exact_matches", "memorization_risk"]].to_string(index=False))


if __name__ == "__main__":
    main()