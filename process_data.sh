  python ./tools/preprocess_data.py \
    --input ./alpaca_data_format.json \
    --tokenizer-name-or-path /mnt/lustrenew/share_data/PAT/datasets/llama2/70B/ \
    --output-prefix ./pdopt/70B80layers \
    --workers 1 \
    --log-interval 1000 \
    --tokenizer-type PretrainedFromHF \
    --chunk-size 1 \
    --json-key output 