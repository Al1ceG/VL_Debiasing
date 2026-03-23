import pandas as pd
import ast
import os

# --- PART 1: YOUR EXISTING QUALITY CHECKS ---
# def classify(x):
#     if pd.isna(x): return 'nan'
#     if not isinstance(x, str): return 'non-string'
#     try:
#         result = ast.literal_eval(x)
#         if isinstance(result, list) and len(result) > 0: return 'good'
#         if isinstance(result, list): return 'empty-list'
#         return f'non-list ({type(result).__name__})'
#     except: return 'parse-error'

# print(">>> RUNNING DATA QUALITY CHECKS...")
# for name, f in [('debiased', 'results/clipcap_debiased.csv'), ('baseline', 'results/clip_cap_baseline.csv')]:
#     if not os.path.exists(f):
#         print(f"Skipping {name}: file not found.")
#         continue
#     df = pd.read_csv(f)
#     counts = df['gt_captions'].apply(classify).value_counts()
#     print(f'\n{name}: {len(df)} total rows')
#     print(counts.to_string())

# --- PART 2: RESEARCH PAPER EXAMPLE FINDER ---
# Set paths
baseline_path = "results/clip_cap_baseline.csv"
debiased_path = "results/clipcap_debiased.csv"
output_file = "results/qualitative_comparison.csv"

def get_first_gt(caption_str):
    """Helper to extract the first human caption for easy reading."""
    try:
        # Converts string-list "[...]" into a real Python list
        captions = ast.literal_eval(caption_str)
        return captions[0] if isinstance(captions, list) and len(captions) > 0 else "N/A"
    except:
        return "N/A"

print(">>> Loading and Merging Results...")
df_base = pd.read_csv(baseline_path)
df_debi = pd.read_csv(debiased_path)

# Determine the unique ID (image_id or file_path)
merge_key = 'image_id' if 'image_id' in df_base.columns else 'file_path'

# Merge the dataframes side-by-side
comparison = pd.merge(
    df_base[[merge_key, 'ground_truth_gender', 'gt_captions', 'generated_text']], 
    df_debi[[merge_key, 'generated_text']], 
    on=merge_key, 
    suffixes=('_baseline', '_debiased')
)

# Find rows where the actual text changed
diff_df = comparison[comparison['generated_text_baseline'] != comparison['generated_text_debiased']].copy()

# Create the "Readable" human caption column
diff_df['human_reference'] = diff_df['gt_captions'].apply(get_first_gt)

# Final Column Selection (Organized for your paper)
final_output = diff_df[[
    merge_key, 
    'ground_truth_gender',
    'human_reference',         
    'generated_text_baseline', 
    'generated_text_debiased', 
    'gt_captions'              
]]

# Save the CSV
if not os.path.exists('results'):
    os.makedirs('results')

final_output.to_csv(output_file, index=False)
print(f"Found {len(final_output)} images with caption changes. Saved to: {output_file}")


# --- PART 3: THE "GENDER CORRECTION" FILTER ---

print("\n>>> FILTERING FOR BEST PAPER EXAMPLES...")

def is_gender_correction(row):
    base = str(row['generated_text_baseline']).lower()
    debi = str(row['generated_text_debiased']).lower()
    gt = str(row['ground_truth_gender']).lower()
    
    # Case 1: GT is Female, Baseline guessed Male, Debiased guessed Female
    if gt == 'female':
        if ('man' in base or 'boy' in base) and ('woman' in debi or 'girl' in debi):
            return True
            
    # Case 2: GT is Male, Baseline guessed Female, Debiased guessed Male
    if gt == 'male':
        if ('woman' in base or 'girl' in base) and ('man' in debi or 'boy' in debi):
            return True
            
    return False

# Apply and Save
best_examples = final_output[final_output.apply(is_gender_correction, axis=1)].copy()
filter_output_file = "results/gender_corrections_only.csv"
best_examples.to_csv(filter_output_file, index=False)

print(f"FOUND {len(best_examples)} PERFECT GENDER CORRECTIONS!")
print(f"These are saved separately to: {filter_output_file}")

if not best_examples.empty:
    print("\n--- PREVIEW OF CORRECTIONS ---")
    print(best_examples[['ground_truth_gender', 'generated_text_baseline', 'generated_text_debiased']].head())