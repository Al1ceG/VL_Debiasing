New-Item -ItemType Directory -Force -Path unified_debiasing/embedding
New-Item -ItemType Directory -Force -Path unified_debiasing/embedding/flickr
New-Item -ItemType Directory -Force -Path unified_debiasing/checkpoint

python unified_debiasing/preprocessing/clip_extract_embedding.py
python unified_debiasing/preprocessing/clipcap_extract_embedding.py