python -m src.data.generate_apr \
    --problems_path data/train_problems.json \
    --output_dir data \
    --max_beam_size 10 \
    --max_sub_call_beam_size 15 \
    --total_samples 500000 \
    --promising_threshold 0.9 \
    --num_workers 64

python -m src.data.generate_sosp \
    --problems_path data/train_problems.json \
    --output_dir data \
    --num_workers 64
