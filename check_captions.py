import pandas as pd
import ast

df1 = pd.read_csv("results/clipcap_debiased.csv")
broken = df1['gt_captions'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x).apply(lambda x: len(x) == 0 if isinstance(x, list) else True)
print("debaised /n", broken.sum())


df2 = pd.read_csv("results/clip_cap_baseline.csv")
broken = df2['gt_captions'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x).apply(lambda x: len(x) == 0 if isinstance(x, list) else True)
print("baseline /n", broken.sum())

def classify(x):
    if pd.isna(x): return 'nan'
    if not isinstance(x, str): return 'non-string'
    try:
        result = ast.literal_eval(x)
        if isinstance(result, list) and len(result) > 0: return 'good'
        if isinstance(result, list): return 'empty-list'
        return f'non-list ({type(result).__name__})'
    except: return 'parse-error'

for name, f in [('debiased', 'results/clipcap_debiased.csv'), ('baseline', 'results/clip_cap_baseline.csv')]:
    df = pd.read_csv(f)
    counts = df['gt_captions'].apply(classify).value_counts()
    print(f'\n{name}: {len(df)} total rows')
    print(counts.to_string())
