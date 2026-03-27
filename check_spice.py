import pandas as pd
import ast

file_to_check = "results/clip_cap_sfid_['decoder']_50_50_0.9_features.csv"
df = pd.read_csv(file_to_check)

def check_gt(val):
    try:
        res = ast.literal_eval(str(val))
        return len(res) if isinstance(res, list) else 0
    except:
        return 0

df['gt_count'] = df['gt_captions'].apply(check_gt)
missing = df[df['gt_count'] == 0]

if not missing.empty:
    print(f"Found {len(missing)} rows with NO ground truth captions!")
    print(missing[['image_id', 'gt_captions']].head())
else:
    print("All rows have ground truth strings, checking for ID formatting...")
    