# predict.py
import argparse
import pandas as pd

from aigt import Detector


def main():
    ap = argparse.ArgumentParser(description="AIGT batch prediction")
    ap.add_argument("--input", "-i", required=True, help="Input dataframe (parquet/csv)")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--id-col", default="prediction_id")
    ap.add_argument("--lang-col", default=None)
    ap.add_argument("--output-prefix", "-o", default="aigt_out")

    args = ap.parse_args()

    # ---- load dataframe ----
    if args.input.endswith(".parquet"):
        df = pd.read_parquet(args.input)
    elif args.input.endswith(".csv"):
        df = pd.read_csv(args.input)
    else:
        raise ValueError("Unsupported input format (use parquet or csv)")

    texts = df[args.text_col].astype(str).tolist()
    doc_ids = df[args.id_col].astype(str).tolist()

    if args.lang_col and args.lang_col in df.columns:
        langs = df[args.lang_col].astype(str).tolist()
    else:
        langs = "en"

    # ---- detector ----
    det = Detector.from_hf(
        repo_id="Trinotrotolueno/aigt-loras",
        subdir_by_lang={
            "en": "en/best",
            "es": "es/best",
            "fr": "fr/best",
            "de": "de/best",
            "it": "it/best",
            "ja": "ja/best",
        },
    )

    articles_df, windows_df = det.predict(
        texts=texts,
        doc_ids=doc_ids,
        lang=langs,
    )

    # ---- save outputs ----
    articles_path = f"{args.output_prefix}_articles.parquet"
    windows_path = f"{args.output_prefix}_windows.parquet"

    articles_df.to_parquet(articles_path, index=False)
    windows_df.to_parquet(windows_path, index=False)

    print(f"[OK] Saved:")
    print(" ", articles_path)
    print(" ", windows_path)


if __name__ == "__main__":
    main()
