import pandas as pd
import json
from pathlib import Path

def load_and_prepare(faq_path, funds_path):
    faqs = pd.read_csv(faq_path)   # expect columns like 'question','answer','id' or combine

    # -----------------------
    # Load funds and normalize column names
    # -----------------------
    funds = pd.read_csv(funds_path)

    # map common CSV column names to the names the rest of the code expects
    rename_map = {
        'cagr_3yr (%)': '3yr_cagr',     # "cagr_3yr (%)" -> "3yr_cagr"
        'cagr_1yr (%)': '1yr_cagr',
        'cagr_5yr (%)': '5yr_cagr',
        'volatility (%)': 'volatility', # remove parentheses
        'sharpe_ratio': 'sharpe_3yr',   # if your CSV has a generic sharpe, map to sharpe_3yr
        # add other mappings if needed
    }
    # apply rename only for columns that exist in the CSV
    existing_map = {k: v for k, v in rename_map.items() if k in funds.columns}
    if existing_map:
        funds = funds.rename(columns=existing_map)

    # keep original data safe (optional)
    # funds_orig = funds.copy()

    corpus_rows = []
    # FAQs -> text
    for i, r in faqs.iterrows():
        text = (str(r.get('question','')) + '\n' + str(r.get('answer',''))).strip()
        corpus_rows.append({
            'id': f'faq_{i}',
            'source_type': 'faq',
            'text': text,
            'metadata': {'raw': r.to_dict()}
        })

    # Funds -> textualized metrics + structured storage
    for i, r in funds.iterrows():
        name = r.get('fund_name') or r.get('Fund') or f"Fund_{i}"
        metrics = []
        # add common metrics if present
        for col in ['1yr_cagr','3yr_cagr','5yr_cagr','volatility','sharpe_3yr','expense_ratio']:
            if col in r and pd.notnull(r[col]):
                metrics.append(f"{col}:{r[col]}")
        desc = r.get('description','')
        text_parts = [f"{name} | " + " | ".join(metrics)]
        if desc:
            text_parts.append("desc: " + str(desc))
        text = " ".join(text_parts)
        corpus_rows.append({
            'id': f'fund_{i}',
            'source_type': 'fund',
            'text': text,
            'metadata': {'fund_name': name, 'raw': r.to_dict()}
        })

    corpus_df = pd.DataFrame(corpus_rows)
    return corpus_df, funds

if __name__=='__main__':
    corpus_df, funds_df = load_and_prepare('/mnt/data/faqs.csv','/mnt/data/funds.csv')
    corpus_df.to_parquet('data/corpus.parquet', index=False)
    funds_df.to_parquet('data/funds_parquet.parquet', index=False)
    print("Saved prepared corpus and funds.")
