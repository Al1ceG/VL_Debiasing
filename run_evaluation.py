"""
run_evaluation.py

Runs evaluation metrics on already-generated caption CSV files.
Does NOT load any CLIP or GPT2 models — only reads CSVs and computes metrics.

All metric computation comes from unified_debiasing/evaluation.py:
    - evaluate_image_captioning() is the main entry point
    - Runs bootstrapping (100 iterations, 1000 samples each) for statistical confidence
    - Outputs mean ± margin at 95% confidence interval for each metric
    - Appends a row to results/eval_results.csv after each file so all experiments are in one place

Metrics computed:
    - METEOR      : caption quality. Requires Java (meteor-1.5.jar runs as a subprocess).
    - CIDEr       : caption quality. Python, fast.
    - BLEU-4      : caption quality. Python, fast.
    - CLIPScore   : image-caption similarity. Requires GPU + image dir. Set COCO_IMG_DIR below to enable.
    - SPICE       : caption quality (scene graph). Requires Java. Set RUN_SPICE=True below to enable.
    - Male/Female/Overall/Composite Misclassification Rate : gender bias metrics.
    - Caption-ABLE: harmonic mean of METEOR and fairness (exp(-MR_C)).

To run:
    python run_evaluation.py
"""

import os
import torch

# evaluate_image_captioning is defined in unified_debiasing/evaluation.py.
# It handles: reading the CSV, optional CLIPScore/SPICE pre-computation, bootstrapping,
# confidence intervals, printing results, and saving to results/eval_results.csv.
from unified_debiasing.evaluation import evaluate_image_captioning


# ── Configuration — edit these as needed ─────────────────────────────────────

# CSV files to evaluate — add or remove paths as needed
RESULTS_FILES = [
    "results/clip_cap_baseline.csv",
    "results/clipcap_debiased.csv",
]

# Set to the COCO val2014 image directory to enable CLIPScore, or None to skip
COCO_IMG_DIR = None  # e.g. "data/COCO/images/val2014"

# Set to True to run SPICE (slow, requires Java, but pre-computed once not 200x)
RUN_SPICE = False

# ─────────────────────────────────────────────────────────────────────────────


def main():
    # GPU setup — falls back to CPU automatically if no GPU available
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Evaluate each CSV in turn.
    # Results are printed to stdout (visible in slurm .out file)
    # and appended as a new row to results/eval_results.csv.
    for file_path in RESULTS_FILES:
        print(f"\n{'='*60}")
        print(f"Evaluating: {file_path}")
        evaluate_image_captioning(
            file_path,
            run_spice=RUN_SPICE,
            coco_img_dir=COCO_IMG_DIR,
            device=device,
        )


if __name__ == "__main__":
    main()
