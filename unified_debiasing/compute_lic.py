import os
import re
import ast
import argparse
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

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
def compute_lic_for_texts(texts, labels, model_path, num_iterations=1):
    scores = []
    # model_path = os.path.expanduser("~/VL_Debiasing/LIC_huggingface")
    model_path = "./LIC_huggingface"
    
    # Suppress heavy logging
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    print(">>> Loading Tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(model_path, local_files_only=True)

    for i in range(num_iterations):
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
            output_dir='./results_temp',          
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

        # LOCAL TRAINING ARGUMENTS
        training_args = TrainingArguments(
        output_dir='./results_temp',          
        num_train_epochs=1,              # Reduce from 3 to 1 for the test
        per_device_train_batch_size=4,   # Reduce from 64 to 4 (Crucial for Mac RAM)
        per_device_eval_batch_size=4,   
        warmup_steps=10,                 # Lower warmup for small test
        weight_decay=0.01,               
        logging_strategy="steps",        # Change "no" to "steps" so you see progress
        logging_steps=10,                # Print progress every 10 steps
        eval_strategy="no",
        save_strategy="no",
        report_to="none",
        )
            
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
    parser.add_argument("--file_path", type=str, required=True, help="Path to the captions CSV file")
    args = parser.parse_args()

    model_path = os.path.expanduser("~/VL_Debiasing/LIC_huggingface")
    print(f"Loading data from {args.file_path}")
    df = pd.read_csv(args.file_path)

    # Filter valid genders and map to binary (Male=0, Female=1)
    df = df[df['ground_truth_gender'].isin(['Male', 'Female'])].copy()
    df['label'] = df['ground_truth_gender'].map({'Male': 0, 'Female': 1})

    # Extract first human caption & mask text
    print("Masking gender words...")
    df['gt_first'] = df['gt_captions'].apply(get_first_gt)
    df['masked_human'] = df['gt_first'].apply(mask_gender_words)
    df['masked_model'] = df['generated_text'].apply(mask_gender_words)

    texts_human = df['masked_human'].tolist()
    texts_model = df['masked_model'].tolist()
    labels = df['label'].tolist()

    #### CHANGE NUM_TERATIONS TO 10 FOR CLUSTER
    print("\n=============================================")
    print("1/2: Computing LIC for HUMAN Captions")
    human_mean, human_std = compute_lic_for_texts(texts_human, labels, model_path, num_iterations=1)    
    print("\n=============================================")
    print("2/2: Computing LIC for MODEL Captions")
    model_mean, model_std = compute_lic_for_texts(texts_model, labels, model_path, num_iterations=1)

    # -----------------------------------------------------------------------------
    # 5. LIC evaluation metrics
    # -----------------------------------------------------------------------------

    # Calculate the core metrics
    bias_amp = model_mean - human_mean
    bias_amp_str = f"{bias_amp:.4f}"
    model_lic_str = f"{model_mean:.4f} ± {model_std:.4f}"
    human_lic_str = f"{human_mean:.4f} ± {human_std:.4f}"

    # FILE 1: The "Deep Dive" (all_lic_results.csv)
    # Includes every number to track calculations
    deep_dive_path = os.path.join(os.path.dirname(args.file_path), "all_lic_results.csv")
    deep_dive_data = {
        'file_path': args.file_path,
        'Human_LIC_Mean': human_mean,
        'Human_LIC_Std': human_std,
        'Model_LIC_Mean': model_mean,
        'Model_LIC_Std': model_std,
        'Bias_Amplification': bias_amp,
        'Human_Full': human_lic_str,
        'Model_Full': model_lic_str
    }
    
    deep_df = pd.DataFrame([deep_dive_data])
    if os.path.exists(deep_dive_path):
        deep_df.to_csv(deep_dive_path, mode='a', header=False, index=False)
    else:
        deep_df.to_csv(deep_dive_path, index=False)

    # FILE 2: The "Master Table" (eval_results_2.csv)
    # Appends only the most important metric (LIC) to the existing evaluation row
    master_path = os.path.join(os.path.dirname(args.file_path), "eval_results_2.csv")

    if os.path.exists(master_path):
        master_df = pd.read_csv(master_path)

        # Match the row by file_path to add the LIC metric
        if args.file_path in master_df['file_path'].values:
            # We use the Model LIC string as the primary metric for the master table
            master_df.loc[master_df['file_path'] == args.file_path, 'LIC'] = model_lic_str
            # Also adding Bias Amp as it's the key indicator of debiasing success
            master_df.loc[master_df['file_path'] == args.file_path, 'Bias_Amp'] = bias_amp_str
            master_df.to_csv(master_path, index=False)
            print(f"Updated master table: {master_path}")
    else:
        print("Warning: eval_results_2.csv not found. Master row update skipped.")

    print(f"\nCalculation Tracked: Model({model_lic_str}) - Human({human_lic_str}) = {bias_amp_str}")