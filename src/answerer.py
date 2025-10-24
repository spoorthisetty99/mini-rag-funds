import re
import pandas as pd

def parse_metric_query(query):
    q = query.lower()
    if 'sharpe' in q:
        period = '3yr' if '3' in q or '3yr' in q or 'three' in q else None
        return {'metric':'sharpe_3yr' if period=='3yr' else 'sharpe'}
    # add rules for cagr, volatility, sharpe (Funds.csv ).
    if 'cagr' in q or 'return' in q:
        if '3' in q: return {'metric':'3yr_cagr'}
        if '5' in q: return {'metric':'5yr_cagr'}
        return {'metric':'1yr_cagr'}

    if 'volatility' in q or 'risk' in q or 'standard deviation' in q or 'fluctuation' in q:
        return {'metric': 'volatility (%)'}
    return None

def rank_funds(funds_df, metric, topk=5, ascending=False):
    if metric not in funds_df.columns:
        return []
    df = funds_df[[ 'fund_name', metric ]].dropna()
    df[metric] = pd.to_numeric(df[metric], errors='coerce')
    df = df.dropna().sort_values(by=metric, ascending=ascending) # ascending False to get largest first
    df = df.head(topk)
    rows = df.to_dict('records')
    return rows

def make_answer_template(query, rank_results=None, retrieved=None):
    if rank_results:
        lines = [f"Top {len(rank_results)} funds for your query:"]
        for r in rank_results:
            lines.append(f"- {r['fund_name']}: {r.get('sharpe_3yr')}")
        return "\n".join(lines)
    # otherwise use retrieved snippets
    snippets = "\n\n".join([f"{s['id']}: {s['text'][:300]}" for s in retrieved[:5]])
    return f"Based on retrieved sources:\n{snippets}"
