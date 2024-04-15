# Download extract.py from https://github.com/facebookresearch/esm/raw/main/scripts/extract.py
python extract.py esm2_t36_3B_UR50D ../astral_40_upper.fa embed --include mean --truncation_seq_length 10000
