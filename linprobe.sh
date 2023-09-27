# CUDA_VISIBLE_DEVICES=6 
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 --master_port='20021' main_linprobe.py \
    --accum_iter 4 \
    --batch_size 256 \
    --model vit_large_patch16 --cls_token\
    --finetune 'temp_dir-large-TCR-1e-2-lamda-3/checkpoint-199.pth'\
    --epochs 90 \
    --blr 0.05  \
    --weight_decay 0.0 \
    --log_dir temp_dir-base-TCR-1e-2-lamda-3\
    --output_dir output_dir-base-linprobe-TCR-1e-2-lamda-3\
    --dist_eval --data_path ../imagenet
