import argparse
import pandas as pd
from pathlib import Path


def load_csv(file_path):
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    return pd.read_csv(file_path)


def clean_text(value):
    if pd.isna(value):
        return ""

    text = str(value)
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text.strip()


def save_jsonl(df, text_column, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cleaned_rows = []

    for _, row in df.iterrows():
        text = clean_text(row[text_column])

        if len(text) > 0:
            cleaned_rows.append({"text": text})

    output_df = pd.DataFrame(cleaned_rows)
    output_df.to_json(output_path, orient="records", lines=True, force_ascii=False)

    print(f"Saved {len(output_df)} rows to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert CSV files to JSONL files for LoRA training.")

    parser.add_argument("--train_file", required=True)
    parser.add_argument("--validation_file", required=True)
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--text_column", default="text")

    args = parser.parse_args()

    train_df = load_csv(args.train_file)
    validation_df = load_csv(args.validation_file)
    test_df = load_csv(args.test_file)

    for name, df in [
        ("train", train_df),
        ("validation", validation_df),
        ("test", test_df)
    ]:
        if args.text_column not in df.columns:
            raise ValueError(
                f"Column '{args.text_column}' not found in {name} file. "
                f"Available columns: {list(df.columns)}"
            )

    save_jsonl(train_df, args.text_column, "data/processed/train.jsonl")
    save_jsonl(validation_df, args.text_column, "data/processed/validation.jsonl")
    save_jsonl(test_df, args.text_column, "data/processed/test.jsonl")

    print("Training files are ready.")


if __name__ == "__main__":
    main()