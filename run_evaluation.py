"""
run_evaluation.py

Runs evaluation metrics on already-generated caption CSV files.
Use this when caption generation has already been done (e.g. results/clip_cap_baseline.csv,
results/clipcap_debiased.csv) and you just want to compute the metrics.

This script does NOT load any CLIP or GPT2 models — it only reads CSVs and computes metrics.

All metric computation comes from unified_debiasing/evaluation.py:
    - evaluate_image_captioning() is the main entry point
    - It runs bootstrapping (100 iterations, 1000 samples each) for statistical confidence
    - Outputs mean ± margin at 95% confidence interval for each metric
    - Appends results to results/eval_results.csv so all experiments are in one place

Metrics computed (all from evaluation.py):
    - METEOR      : caption quality. Requires Java (meteor-1.5.jar via subprocess).
    - CIDEr       : caption quality. Pure Python, fast.
    - BLEU-4      : caption quality. Pure Python, fast.
    - CLIPScore   : reference-free image-caption similarity. Requires GPU. Optional (needs --image_dir).
    - SPICE       : caption quality (scene graph). Requires Java. Optional (--run_spice). Slow even pre-computed.
    - Male/Female/Overall/Composite Misclassification Rate : gender bias metrics, from misclassification_rate() in evaluation.py
    - Caption-ABLE: harmonic mean of METEOR and fairness term exp(-MR_C), from evaluation.py

Usage examples:
    # Fast (no CLIPScore, no SPICE):
    python run_evaluation.py --results_files results/clip_cap_baseline.csv results/clipcap_debiased.csv

    # With CLIPScore (slower, needs GPU and image dir):
    python run_evaluation.py --results_files results/clip_cap_baseline.csv --image_dir data/COCO/images/val2014

    # With SPICE (slow, needs Java):
    python run_evaluation.py --results_files results/clip_cap_baseline.csv --run_spice
"""

import argparse
import os
import torch

# evaluate_image_captioning is the main evaluation function defined in unified_debiasing/evaluation.py.
# It handles: reading the CSV, pre-computing CLIPScore/SPICE if requested, bootstrapping,
# confidence interval calculation, printing results, and saving to eval_results.csv.
from unified_debiasing.evaluation import evaluate_image_captioning


def main():
    parser = argparse.ArgumentParser(description="Run evaluation on existing caption CSV files.")

    # One or more CSV files to evaluate — e.g. baseline and debiased results
    parser.add_argument(
        '--results_files', nargs='+', required=True,
        help='Paths to caption CSV files to evaluate (e.g. results/clip_cap_baseline.csv)'
    )

    # Optional: directory of COCO val2014 images, needed to compute CLIPScore.
    # If not provided, CLIPScore is skipped. See compute_clip_scores_per_image() in evaluation.py.
    parser.add_argument(
        '--image_dir', default=None, type=str,
        help='COCO val2014 image directory (optional, required for CLIPScore)'
    )

    # Optional: run SPICE metric. Pre-computed once before bootstrap (see compute_spice_scores_per_image()
    # in evaluation.py). Still slow due to Java scene graph parsing, but much faster than the old
    # approach of running it 200x inside the bootstrap loop.
    parser.add_argument(
        '--run_spice', action='store_true',
        help='Run SPICE metric (requires Java, slow even pre-computed)'
    )

    parser.add_argument('--gpu_id', default='0', type=str, help='GPU id to use')
    args = parser.parse_args()

    # Set up GPU — same pattern as measure_caption_bias.py
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Evaluate each CSV file in turn.
    # Results for each file are printed to stdout (visible in slurm .out)
    # and appended to eval_results.csv in the same directory as the input CSV.
    for file_path in args.results_files:
        print(f"\n{'='*60}")
        print(f"Evaluating: {file_path}")
        evaluate_image_captioning(
            file_path,
            run_spice=args.run_spice,
            coco_img_dir=args.image_dir,
            device=device,
        )


if __name__ == "__main__":
    main()
