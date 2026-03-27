import pandas as pd
import ast
import os

# # --- PART 1: YOUR EXISTING QUALITY CHECKS ---
# # def classify(x):
# #     if pd.isna(x): return 'nan'
# #     if not isinstance(x, str): return 'non-string'
# #     try:
# #         result = ast.literal_eval(x)
# #         if isinstance(result, list) and len(result) > 0: return 'good'
# #         if isinstance(result, list): return 'empty-list'
# #         return f'non-list ({type(result).__name__})'
# #     except: return 'parse-error'

# # print(">>> RUNNING DATA QUALITY CHECKS...")
# # for name, f in [('debiased', 'results/clipcap_debiased.csv'), ('baseline', 'results/clip_cap_baseline.csv')]:
# #     if not os.path.exists(f):
# #         print(f"Skipping {name}: file not found.")
# #         continue
# #     df = pd.read_csv(f)
# #     counts = df['gt_captions'].apply(classify).value_counts()
# #     print(f'\n{name}: {len(df)} total rows')
# #     print(counts.to_string())

# # --- PART 2: RESEARCH PAPER EXAMPLE FINDER ---
# # Set paths
# baseline_path = "results/clip_cap_baseline.csv"
# debiased_path = "results/clipcap_debiased.csv"
# output_file = "results/qualitative_comparison.csv"

# def get_first_gt(caption_str):
#     """Helper to extract the first human caption for easy reading."""
#     try:
#         # Converts string-list "[...]" into a real Python list
#         captions = ast.literal_eval(caption_str)
#         return captions[0] if isinstance(captions, list) and len(captions) > 0 else "N/A"
#     except:
#         return "N/A"

# print(">>> Loading and Merging Results...")
# df_base = pd.read_csv(baseline_path)
# df_debi = pd.read_csv(debiased_path)

# # Determine the unique ID (image_id or file_path)
# merge_key = 'image_id' if 'image_id' in df_base.columns else 'file_path'

# # Merge the dataframes side-by-side
# comparison = pd.merge(
#     df_base[[merge_key, 'ground_truth_gender', 'gt_captions', 'generated_text']], 
#     df_debi[[merge_key, 'generated_text']], 
#     on=merge_key, 
#     suffixes=('_baseline', '_debiased')
# )

# # Find rows where the actual text changed
# diff_df = comparison[comparison['generated_text_baseline'] != comparison['generated_text_debiased']].copy()

# # Create the "Readable" human caption column
# diff_df['human_reference'] = diff_df['gt_captions'].apply(get_first_gt)

# # Final Column Selection (Organized for your paper)
# final_output = diff_df[[
#     merge_key, 
#     'ground_truth_gender',
#     'human_reference',         
#     'generated_text_baseline', 
#     'generated_text_debiased', 
#     'gt_captions'              
# ]]

# # Save the CSV
# if not os.path.exists('results'):
#     os.makedirs('results')

# final_output.to_csv(output_file, index=False)
# print(f"Found {len(final_output)} images with caption changes. Saved to: {output_file}")


# # --- PART 3: THE "GENDER CORRECTION" FILTER ---

# print("\n>>> FILTERING FOR BEST PAPER EXAMPLES...")

# def is_gender_correction(row):
#     base = str(row['generated_text_baseline']).lower()
#     debi = str(row['generated_text_debiased']).lower()
#     gt = str(row['ground_truth_gender']).lower()
    
#     # Case 1: GT is Female, Baseline guessed Male, Debiased guessed Female
#     if gt == 'female':
#         if ('man' in base or 'boy' in base) and ('woman' in debi or 'girl' in debi):
#             return True
            
#     # Case 2: GT is Male, Baseline guessed Female, Debiased guessed Male
#     if gt == 'male':
#         if ('woman' in base or 'girl' in base) and ('man' in debi or 'boy' in debi):
#             return True
            
#     return False

# # Apply and Save
# best_examples = final_output[final_output.apply(is_gender_correction, axis=1)].copy()
# filter_output_file = "results/gender_corrections_only.csv"
# best_examples.to_csv(filter_output_file, index=False)

# print(f"FOUND {len(best_examples)} PERFECT GENDER CORRECTIONS!")
# print(f"These are saved separately to: {filter_output_file}")

# if not best_examples.empty:
#     print("\n--- PREVIEW OF CORRECTIONS ---")
#     print(best_examples[['ground_truth_gender', 'generated_text_baseline', 'generated_text_debiased']].head())


### with all 5 csv files 

# 1. DEFINE YOUR FILES
files = {
    'baseline': "results/clip_cap_baseline.csv",
    'debiased': "results/clipcap_debiased.csv",
    'sfid': "results/clipcap_debiased_sfid_debiased.csv",
    'decoder_only': "results/clip_cap_sfid_['decoder']_50_50_0.9_features.csv",
    'enc_dec': "results/clip_cap_sfid_['decoder', 'encoder']_50_50_0.9_features.csv"
}

def get_first_gt(caption_str):
    try:
        captions = ast.literal_eval(caption_str)
        return captions[0] if isinstance(captions, list) and len(captions) > 0 else "N/A"
    except: return "N/A"

# 2. LOAD AND MERGE ALL 5
print(">>> Merging 5 Results Files...")
master_df = None

for name, path in files.items():
    if not os.path.exists(path):
        print(f"Warning: {path} not found. Skipping.")
        continue
    
    current_df = pd.read_csv(path)
    # Ensure ID is string to avoid merge issues
    merge_key = 'image_id' if 'image_id' in current_df.columns else 'file_path'
    current_df[merge_key] = current_df[merge_key].astype(str)
    
    # Keep only the essential columns and rename the caption column
    current_df = current_df[[merge_key, 'ground_truth_gender', 'gt_captions', 'generated_text']]
    current_df = current_df.rename(columns={'generated_text': f'text_{name}'})
    
    if master_df is None:
        master_df = current_df
    else:
        # Merge only on the ID (we drop duplicate gender/gt columns)
        master_df = pd.merge(master_df, current_df[[merge_key, f'text_{name}']], on=merge_key)

# 3. CALCULATE "DISCREPANCY SCORE" (The sorting magic)
print(">>> Calculating Discrepancy Scores...")

def calculate_discrepancy(row):
    captions = [str(row[f'text_{name}']).lower() for name in files.keys() if f'text_{name}' in row]
    
    # A: Lexical Diversity (How different are the words?)
    # We split into words and see how many unique words exist across all 5 captions
    all_words = set()
    for cap in captions:
        all_words.update(cap.split())
    
    # B: Gender Disagreement (Did any model flip the gender?)
    genders_found = set()
    for cap in captions:
        if any(w in cap for w in ['man', 'boy', 'he', 'his']): genders_found.add('male')
        if any(w in cap for w in ['woman', 'girl', 'she', 'her']): genders_found.add('female')
    
    # SCORE: Number of unique words + (Bonus if gender was flipped)
    score = len(all_words) / 10  # Normalize lexical diff
    if len(genders_found) > 1:
        score += 10 # Massive boost for rows where models disagree on gender
        
    return score

master_df['discrepancy_score'] = master_df.apply(calculate_discrepancy, axis=1)
master_df['human_reference'] = master_df['gt_captions'].apply(get_first_gt)

# 4. SORT AND SAVE
# We put the highest discrepancy at the top (these are the best examples)
master_df = master_df.sort_values(by='discrepancy_score', ascending=False)

# Reorder columns for readability
cols_to_keep = [
    'image_id' if 'image_id' in master_df.columns else 'file_path',
    'ground_truth_gender',
    'discrepancy_score',
    'human_reference',
    'text_baseline',
    'text_debiased',
    'text_sfid',
    'text_decoder_only',
    'text_enc_dec'
]
master_df = master_df[cols_to_keep]

master_df.to_csv("results/five_way_comparison.csv", index=False)
print(f"\nDONE! Saved 5-way comparison to results/five_way_comparison.csv")
print(f"The top 10 rows in this file are your 'best' qualitative examples.")