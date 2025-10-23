# src/preprocessing.py
import pandas as pd
from pathlib import Path
import numpy as np
import uuid

def _make_id(prefix: str, i: int):
    return f"{prefix}-{i}"

def load_and_prepare(faq_path: str, funds_path: str):
    """
    Loads faqs.csv and funds.csv and returns:
      - corpus_df: DataFrame with columns ['id', 'text', 'metadata']
        metadata is a dict containing at least: {'source_file': <str>, 'origin_id': <original row id>}
      - funds_df: original funds dataframe (kept for metric ranking)
    """
    faq_path = Path(faq_path)
    funds_path = Path(funds_path)

    # load FAQs
    faqs = pd.read_csv(str(faq_path))
    # normalize columns that may exist
    if "question" in faqs.columns and "answer" in faqs.columns:
        faqs["text"] = faqs["question"].fillna("") + ". " + faqs["answer"].fillna("")
    elif "text" in faqs.columns:
        faqs["text"] = faqs["text"].fillna("")
    else:
        # if structure unknown, coalesce all columns into a text field
        faqs["text"] = faqs.fillna("").astype(str).agg(" ".join, axis=1)

    # create corpus rows for faqs
    faq_rows = []
    for i, row in faqs.reset_index(drop=True).iterrows():
        origin_id = row.get("id", i)
        uid = _make_id("faq", i)
        faq_rows.append({
            "id": uid,
            "text": str(row["text"]),
            "metadata": {
                "source_file": faq_path.name,
                "origin_id": origin_id,
                # you can add more contextual fields here if desired (e.g., 'question' or 'title')
            }
        })

    # load Funds
    funds = pd.read_csv(str(funds_path))
    # ensure some numeric columns exist (CAGR, sharpe, volatility) - don't enforce names but try common variants
    # Build textual description for each fund row
    fund_rows = []
    for i, row in funds.reset_index(drop=True).iterrows():
        origin_id = row.get("id", i)
        fund_name = row.get("fund_name") or row.get("name") or row.get("fund") or ""
        # try to collect a few numeric metrics if present
        metrics = []
        # common metric column patterns - adjust as needed for your dataset
        for col in ["3yr_cagr", "3_year_cagr", "cagr_3y", "3yr_return", "three_year_return"]:
            if col in funds.columns:
                metrics.append(f"3yr_CAGR = {row[col]}")
                break
        for col in ["sharpe_3yr", "3yr_sharpe", "sharpe"]:
            if col in funds.columns:
                metrics.append(f"Sharpe_3yr = {row[col]}")
                break
        for col in ["volatility_1yr", "1yr_volatility", "volatility"]:
            if col in funds.columns:
                metrics.append(f"Volatility_1yr = {row[col]}")
                break

        desc_parts = [f"Fund: {fund_name}"] if fund_name else []
        if len(metrics):
            desc_parts += metrics
        # fallback: concat non-empty string fields
        if not desc_parts:
            # take first 5 columns as textual fallback
            text_fallback = " ".join([str(row[c]) for c in funds.columns[:5]])
            desc = f"{fund_name} {text_fallback}"
        else:
            desc = ". ".join(desc_parts)

        uid = _make_id("fund", i)
        fund_rows.append({
            "id": uid,
            "text": desc,
            "metadata": {
                "source_file": funds_path.name,
                "origin_id": origin_id,
                "fund_name": fund_name
            }
        })

    # combine into corpus
    combined = faq_rows + fund_rows
    corpus_df = pd.DataFrame(combined)
    # ensure columns exist
    corpus_df = corpus_df[["id", "text", "metadata"]]

    # return corpus and original funds df (for metric ranking)
    return corpus_df, funds
