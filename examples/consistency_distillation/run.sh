export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="./out"

accelerate launch train_lcm_distill_sd_wds.py \
    --pretrained_teacher_model=$MODEL_NAME \
    --output_dir=$OUTPUT_DIR \
    --mixed_precision=fp16 \
    --resolution=512 \
    --learning_rate=1e-6 --loss_type="huber" --ema_decay=0.95 --adam_weight_decay=0.0 \
    --max_train_steps=20000 \
    --max_train_samples=4000000 \
    --dataloader_num_workers=8 \
    --train_shards_path_or_url="pipe:curl -L -s https://huggingface.co/datasets/laion/conceptual-captions-12m-webdataset/resolve/main/data/{00000..01099}.tar?download=true" \
    --validation_steps=200 \
    --checkpointing_steps=200 --checkpoints_total_limit=10 \
    --train_batch_size=12 \
    --gradient_checkpointing \
    --gradient_accumulation_steps=1 \
    --resume_from_checkpoint=latest \
    --seed=453645634
