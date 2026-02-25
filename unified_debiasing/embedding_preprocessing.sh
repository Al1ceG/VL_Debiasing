#!/bin/bash\
mkdir unified_debiasing/embedding
# mkdir unified_debiasing/embedding/flickr
mkdir unified_debiasing/checkpoint


# Extract debiasing embedding for zero-shot classification and text-to-image retrieval
# python unified_debiasing/preprocessing/clip_extract_embedding.py
python unified_debiasing/preprocessing/facet_extract_embedding.py
# python unified_debiasing/preprocessing/flickr_extract_embedding.py

# # Extract debiasing embedding for image captioning
# python unified_debiasing/preprocessing/clipcap_extract_embedding.py
# python unified_debiasing/preprocessing/blip_extract_embedding.py

# # Extract debiasing embedding for text-to-image generation
# python unified_debiasing/preprocessing/codi_extract_embedding.py
# python unified_debiasing/preprocessing/sd_extract_embedding.py




