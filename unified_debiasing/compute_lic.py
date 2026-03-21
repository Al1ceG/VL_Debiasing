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
# 1. Masking Logic (Exactly from Hendricks et al. / Hirota et al.)
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
# 2. PyTorch Dataset
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
# 3. BERT Training Loop (10 iterations) - ON CLUSTER
# -----------------------------------------------------------------------------
def compute_lic_for_texts(texts, labels, num_iterations=10):
    scores = []
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') # convert text to ids for BERT
    
    # Suppress heavy logging
    os.environ["WANDB_DISABLED"] = "true"

    for i in range(num_iterations):
        print(f"\n--- Iteration {i+1}/{num_iterations} ---")
        # 80/20 train/test split
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=i
        )
        
        train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=64)
        test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=64) # max tokens = 64
        
        train_dataset = CaptionDataset(train_encodings, train_labels)
        test_dataset = CaptionDataset(test_encodings, test_labels)
        
        #download fresh BERT model, (2 choices M, F), unbiased model every iteration
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
        
        training_args = TrainingArguments(
            output_dir='./results_temp',          
            num_train_epochs=3,              
            per_device_train_batch_size=64,  # Takes advantage of your 48GB GPU
            per_device_eval_batch_size=64,   
            warmup_steps=100,                
            weight_decay=0.01,               
            logging_strategy="no",
            evaluation_strategy="no",
            save_strategy="no",
            report_to="none"
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
        
    return np.mean(scores), np.std(scores)

# -----------------------------------------------------------------------------
# 4. Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True, help="Path to the captions CSV file")
    args = parser.parse_args()

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

    print("\n=============================================")
    print("1/2: Computing LIC for HUMAN Captions")
    print("=============================================")
    human_mean, human_std = compute_lic_for_texts(texts_human, labels, num_iterations=10)
    
    print("\n=============================================")
    print("2/2: Computing LIC for MODEL Captions")
    print("=============================================")
    model_mean, model_std = compute_lic_for_texts(texts_model, labels, num_iterations=10)

    # Bias Amplification = Model LIC - Human LIC
    bias_amp = model_mean - human_mean
    
    print(f"\nFinal Results for {args.file_path}:")
    print(f"Human LIC: {human_mean:.4f} ± {human_std:.4f}")
    print(f"Model LIC: {model_mean:.4f} ± {model_std:.4f}")
    print(f"Bias Amplification: {bias_amp:.4f}")

    lic_path = args.file_path.replace('.csv', '') + '_lic.csv'
    pd.DataFrame({'LIC': [f"{bias_amp:.4f}"]}).to_csv(lic_path, index=False)
    print(f"\nSaved Bias Amplification to {lic_path}")

    # Save detailed results
    results_path = args.file_path.replace('.csv', '') + '_lic_details.csv'
    pd.DataFrame({
        'Human LIC Mean': [human_mean],
        'Human LIC Std': [human_std],
        'Model LIC Mean': [model_mean],
        'Model LIC Std': [model_std],
        'Bias Amplification': [bias_amp]
    }).to_csv(results_path, index=False)
    print(f"Saved detailed results to {results_path}")