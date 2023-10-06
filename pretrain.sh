python -m torch.distributed.launch --nproc_per_node=8 --master_port='30111' main_pretrain.py \
    --batch_size 128 \
    --model mae_vit_large_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 200 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path ../imagenet\
    --lamb 1e-2\
    --uniformity_lamda 3\
    --reg TCR\
    --centering False\
    --output_dir temp_dir-large-TCR-1e-2-lamda-3\
    --log_dir temp_dir-large-TCR-1e-2-lamda-3\
    # --distributed
