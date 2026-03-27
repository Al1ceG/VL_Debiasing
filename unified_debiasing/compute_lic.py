import os
import re
import ast
import argparse
import pandas as pd
import numpy as np
import torch
import random
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datetime import datetime

# -----------------------------------------------------------------------------
# Masking Logic (Exactly from Hendricks et al. / Hirota et al.)
# -----------------------------------------------------------------------------
MALE_WORDS = ['man', 'men', 'boy', 'boys', 'he', 'his', 'him', 'himself', 'gentleman', 'gentlemen', 'male', 'males', 'guy', 'guys', 'father', 'fathers', 'son', 'sons', 'brother', 'brothers', 'husband', 'husbands', 'uncle', 'uncles']
FEMALE_WORDS = ['woman', 'women', 'girl', 'girls', 'she', 'her', 'hers', 'herself', 'lady', 'ladies', 'female', 'females', 'mother', 'mothers', 'daughter', 'daughters', 'sister', 'sisters', 'wife', 'wives', 'aunt', 'aunts']
ALL_GENDERED_WORDS = MALE_WORDS + FEMALE_WORDS

PATTERN = r'\b(' + '|'.join(ALL_GENDERED_WORDS) + r')\b'



def mask_gender_words(text):
    """Replaces explicit gender words with the BERT [MASK] token."""
    if not isinstance(text, str):
        return ""
    return re.sub(PATTERN, '[MASK]', text, flags=re.IGNORECASE)

def get_first_gt(caption_str):
    """Extracts the first human ground truth caption from the string-list."""
    try:
        captions = ast.literal_eval(caption_str)
        if isinstance(captions, list) and len(captions) > 0:
            return str(captions[0])
    except:
        pass
    return ""

# -----------------------------------------------------------------------------
# PyTorch Dataset
# -----------------------------------------------------------------------------
class CaptionDataset(torch.utils.data.Dataset):
    """Inherit pytorch encodings and labels"""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# -----------------------------------------------------------------------------
# 3. BERT Training Loop (10 iterations) - ON CLUSTER - 1(locally, also batch size 1)
# -----------------------------------------------------------------------------
def compute_lic_for_texts(texts, labels, model_path, num_iterations=10):
    scores = []
    # model_path = os.path.expanduser("~/VL_Debiasing/LIC_huggingface")
    
    # Suppress heavy logging
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    print(">>> Loading Tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(model_path, local_files_only=True)

    for i in range(num_iterations):
        # Reset all seeds at the start of each iteration
        # Each iteration gets a different but fixed seed, so the BERT initialisation is unique 
        # per iteration but identical every time you re-run the script
        random.seed(42 + i)
        np.random.seed(42 + i)
        torch.manual_seed(42 + i)
        torch.cuda.manual_seed_all(42 + i)
    
        print(f"\n--- Iteration {i+1}/{num_iterations} ---")
        # 80/20 train/test split
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=i
        )
        print(">>> Encoding Data...")
        train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=64)
        test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=64) # max tokens = 64
        
        train_dataset = CaptionDataset(train_encodings, train_labels)
        test_dataset = CaptionDataset(test_encodings, test_labels)
        
        #BERT model, (2 choices M, F), unbiased model every iteration
        print(">>> Initializing BERT Model...")
        model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2, local_files_only=True)

        # CLUSTER TRAINING ARGUMENTS
        training_args = TrainingArguments(
            output_dir='./.results_temp',          
            num_train_epochs=3,              
            per_device_train_batch_size=64,  # Takes advantage of your 48GB GPU
            per_device_eval_batch_size=64,
            warmup_steps=100,                
            weight_decay=0.01,               
            logging_strategy="no",
            eval_strategy="no",
            save_strategy="no",
            report_to="none"
        )

        # # LOCAL TRAINING ARGUMENTS
        # training_args = TrainingArguments(
        # output_dir='./results_temp',          
        # num_train_epochs=1,              # Reduce from 3 to 1 for the test
        # per_device_train_batch_size=4,   # Reduce from 64 to 4 (Crucial for Mac RAM)
        # per_device_eval_batch_size=4,   
        # warmup_steps=10,                 # Lower warmup for small test
        # weight_decay=0.01,               
        # logging_strategy="steps",        # Change "no" to "steps" so you see progress
        # logging_steps=10,                # Print progress every 10 steps
        # eval_strategy="no",
        # save_strategy="no",
        # report_to="none",
        # )
            
        trainer = Trainer(
            model=model,                         
            args=training_args,                  
            train_dataset=train_dataset,         
            eval_dataset=test_dataset             
        )
        
        trainer.train()
        
        # Evaluate on test set
        predictions = trainer.predict(test_dataset) # take 20% data, predict gender 
        preds = np.argmax(predictions.predictions, axis=-1)
        accuracy = (preds == test_labels).mean()
        
        print(f"Accuracy for iteration {i+1}: {accuracy:.4f}")
        scores.append(accuracy)

        # clear the GPU after each iteration
        del model
        del trainer
        torch.cuda.empty_cache()
        
    return np.mean(scores), np.std(scores)

# -----------------------------------------------------------------------------
# 4. Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True)
    args = parser.parse_args()

    # Model Path Setup
    model_path = os.path.expanduser("~/VLD/VL_Debiasing/LIC_huggingface")    
    
    # Load Data
    print(f"\n>>> PROCESSING: {args.file_path}")
    df = pd.read_csv(args.file_path)
    df = df[df['ground_truth_gender'].isin(['Male', 'Female'])].copy()
    df['label'] = df['ground_truth_gender'].map({'Male': 0, 'Female': 1})

    df['gt_first'] = df['gt_captions'].apply(get_first_gt)
    df['masked_human'] = df['gt_first'].apply(mask_gender_words)
    df['masked_model'] = df['generated_text'].apply(mask_gender_words)

    # 1/2: Human LIC
    print("\nStarting Human LIC calculation...")
    h_mean, h_std = compute_lic_for_texts(df['masked_human'].tolist(), df['label'].tolist(), model_path, 10)    
    
    # 2/2: Model LIC
    print("\nStarting Model LIC calculation...")
    m_mean, m_std = compute_lic_for_texts(df['masked_model'].tolist(), df['label'].tolist(), model_path, 10)

    # 1. Prepare the strings for saving (Fixed variable names)
    current_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    bias_amp = m_mean - h_mean
    
    # We create these strings so they can be used in both CSV files
    human_lic_str = f"{h_mean:.4f} ± {h_std:.4f}"
    model_lic_str = f"{m_mean:.4f} ± {m_std:.4f}"
    bias_amp_str = f"{bias_amp:.4f}"

    # 2. SAVE RESULTS (Deep Dive CSV)
    output_dir = os.path.dirname(args.file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    deep_dive_path = os.path.join(output_dir, "all_lic_results.csv")
    res_df = pd.DataFrame([{
        'File': os.path.basename(args.file_path),
        'Human_LIC': human_lic_str,
        'Model_LIC': model_lic_str,
        'Bias_Amp': bias_amp_str,
        'Timestamp': current_ts
    }])
    res_df.to_csv(deep_dive_path, mode='a', header=not os.path.exists(deep_dive_path), index=False)

    # 3. PRINT SUMMARY BOX
    print("\n" + "#"*60)
    print(f" FINAL METRICS: {os.path.basename(args.file_path)}")
    print(f" Calculated at: {current_ts}")
    print("#"*60)
    print(f"  > Human LIC:         {human_lic_str}")
    print(f"  > Model LIC:         {model_lic_str}")
    print(f"  > Bias Amplification: {bias_amp_str}")
    print("#"*60 + "\n")

    # -------------------------------------------------------------------------
    # 5. Update Master Table (eval_results_2.csv) - update existing row
    # -------------------------------------------------------------------------
    master_path = os.path.join(output_dir, "eval_results_3_b.csv")

    if os.path.exists(master_path):
        master_df = pd.read_csv(master_path)
        
        # 1. FIX: Convert columns to 'object' so they can hold text like "±" or "N/A"
        for col in ['LIC', 'Bias_Amp', 'Timestamp', 'Notes']:
            if col not in master_df.columns:
                master_df[col] = "N/A"
            master_df[col] = master_df[col].astype(object)

        # 2. Match based on the file path (e.g., "results/clip_cap_baseline.csv")
        target_match = args.file_path 
        
        if target_match in master_df['file_path'].values:
            # CASE A: UPDATE existing row
            idx = master_df['file_path'] == target_match
            master_df.loc[idx, 'LIC'] = model_lic_str
            master_df.loc[idx, 'Bias_Amp'] = bias_amp_str
            master_df.loc[idx, 'Timestamp'] = current_ts
            master_df.loc[idx, 'Notes'] = "Updated LIC metrics"
            print(f"Updated existing row for {target_match}")
        else:
            # CASE B: APPEND new row if not found
            new_row = {
                'file_path': target_match,
                'LIC': model_lic_str,
                'Bias_Amp': bias_amp_str,
                'Timestamp': current_ts,
                'Notes': f"New entry added during LIC run {target_match}"
            }
            master_df = pd.concat([master_df, pd.DataFrame([new_row])], ignore_index=True)
            print(f" Added brand new row for {target_match}")
            
        # 3. Save the final result
        master_df.to_csv(master_path, index=False)

    else:
        # CASE C: CREATE the file if it doesn't exist at all
        print(f"Creating new master table at {master_path}")
        new_df = pd.DataFrame([{
            'file_path': args.file_path,
            'LIC': model_lic_str,
            'Bias_Amp': bias_amp_str,
            'Timestamp': current_ts,
            'Notes': "Initial Master Table Creation"
        }])
        new_df.to_csv(master_path, index=False)