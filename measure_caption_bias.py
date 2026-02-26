import argparse
import os
import sys
import json
import random
import pickle

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import nltk

from transformers import GPT2Tokenizer

# import things from the unified paper
from clip_debiasing.models.clipcap import model_clipcap
from clip_debiasing.models.clipcap.clipcap_utils import decide_gender, generate
import clip
from unified_debiasing.evaluation import evaluate_image_captioning


def main():
    parser = argparse.ArgumentParser(description="Measure baseline bias of ClipCap (no debiasing).")
    parser.add_argument('--gpu_id', default='0', type=str, help='GPU id to use')
    parser.add_argument(
        '--image_dir',
        default="data/COCO/images/val2014",
        type=str,
        help='Directory containing COCO val2014 images',
    )

    parser.add_argument(
        '--results_filename',
        default="results/clip_cap_baseline.csv",
        type=str,
        help='Path to save baseline captioning results',
    )
    args = parser.parse_args()

    # Set CUDA device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Ensure imports can find project root
    sys.path.append('./')

    # Ensure required NLTK resources are available
    nltk.download('punkt')
    nltk.download('punkt_tab')

    # Reproducibility
    seed = 0
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmarks = False
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Load ClipCap model and CLIP image encoder
    prefix_length = 10
    model_path = 'clip_debiasing/models/clipcap/clip_cap_coco_weight.pt'
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = model_clipcap.ClipCaptionModel(prefix_length, device=device)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)

    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    model = model.eval().to(device)

    # Load COCO captions annotations
    with open('data/COCO/annotations/captions_val2014.json', 'r') as json_data:
        d = json.load(json_data)
    annotations = d['annotations']

    # Map image_id -> list of ground-truth captions
    id_to_captions = {}
    for ann in annotations:
        image_id = ann['image_id']
        caption = ann['caption']
        if image_id not in id_to_captions:
            id_to_captions[image_id] = []
        id_to_captions[image_id].append(caption)

    # TODO: not just for gender
    # Ground-truth gender per image
    imid_2_gender = pickle.load(open('clip_debiasing/models/clipcap/val_imid_gender.pkl', 'rb'))
    filtered_image_ids = set(imid_2_gender.keys())

    # Only keep images that have a gender label
    filtered_id_to_captions = {
        image_id: captions
        for image_id, captions in id_to_captions.items()
        if image_id in filtered_image_ids
    }

    # IDs to remove (e.g., problematic images)
    remove_id = pd.read_csv("clip_debiasing/models/clipcap/remove_df.csv")['remove_id']

    results_filename = args.results_filename
    results = []

    # Only run generation if file does not already exist
    if not os.path.exists(results_filename):
        print("Results file does not exist")
        for image_id, gt_captions in tqdm(filtered_id_to_captions.items()):
            with torch.no_grad():
                if image_id in remove_id.values:
                    continue

                image_path = os.path.join(
                    args.image_dir,
                    f"COCO_val2014_{str(image_id).zfill(12)}.jpg",
                )
                image = Image.open(image_path).convert('RGB')
                ground_truth_gender = imid_2_gender[image_id]

                # Get CLIP image embedding (prefix)
                prefix = clip_model.encode_image(
                    preprocess(image).unsqueeze(0).to(device)
                ).float()

                # Generate caption without any debiasing
                generated_text = generate(
                    model,
                    tokenizer,
                    embed=prefix,
                )

                # Detect gender in generated text
                detected_gender = decide_gender(nltk.word_tokenize(generated_text))

                # Store result
                results.append(
                    {
                        'image_id': image_id,
                        'ground_truth_gender': ground_truth_gender,
                        'detected_gender': detected_gender,
                        'gt_captions': gt_captions,
                        'generated_text': generated_text,
                    }
                )

        df = pd.DataFrame(results)
        os.makedirs(os.path.dirname(results_filename), exist_ok=True)
        df.to_csv(results_filename, index=False)
    else:
        print(f"Results file {results_filename} already exists. Skipping generation.")

    # Evaluate baseline image captioning and bias
    evaluate_image_captioning(results_filename)


if __name__ == "__main__":
    main()

