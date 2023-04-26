# pretrain
python3 -m torch.distributed.launch --nproc_per_node=8 \
        --node_rank 0 \
        --master_addr=${MASTER_ADDR:-127.0.0.1} \
        --master_port=${MASTER_PORT:-46123} \
        ./main_pos_bot.py \
        --arch pos_small \
        --output_dir ./output_100eps_005/ \
        --data_path /earth-nas/datasets/imagenet-1k/train/ \
        --world_size 8 \
        --local_crops_number 1 \
        --local_crops_scale 0.05 0.25 \
        --global_crops_scale 0.25 1 \
        --pred_ratio 0 0.3 \
        --norm_last_layer false \
        --shared_head true \
        --pred_ratio_var 0 0.2 \
        --lambda3 1.0 \
        --batch_size_per_gpu 64 \
        --lambda2 1.0 \
        --epochs 100 \
        --warmup_teacher_temp_epochs 30 \
        --teacher_query_temp 0.07 \
        --teacher_temp 0.07 \
        --local_crops_size 96

# linear eval
python3 -m torch.distributed.launch --nproc_per_node=8 \
        --master_port=${MASTER_PORT:-36124} \
        ./evaluation/eval_linear.py \
            --pretrained_weights ./output_400eps_base/checkpoint0360.pth --n_last_blocks 4 \
            --avgpool_patchtokens 0     --arch pos_base     --checkpoint_key teacher \
            --output_dir ./evaluation/output_cls/ \
            --dist_url tcp://localhost:23142 \
            --data_path /earth-nas/datasets/imagenet-1k/

# KNN
python3 -m torch.distributed.launch --nproc_per_node=8 \
        --master_port=${MASTER_PORT:-23921} \
        ./evaluation/eval_knn.py \
            --pretrained_weights ./output_100eps_020/checkpoint.pth --n_last_blocks 4 \
            --avgpool_patchtokens 0     --arch pos_small     --checkpoint_key teacher \
            --dist_url tcp://localhost:23142 \
            --data_path /earth-nas/datasets/imagenet-1k/

# fine tune
python3 -m torch.distributed.launch --nproc_per_node=8 \
                --master_port=${MASTER_PORT:-26361} \
                ./evaluation/classification_layer_decay/run_class_finetuning.py \
                --finetune ./output_ckpt/pretrain_vit_base.pth \
                --model vit_base \
                --epochs 100 \
                --warmup_epochs 5 \
                --layer_decay 0.65 \
                --mixup 0.8 \
                --lr 5e-4 \
                --cutmix 1.0 \
                --drop_path 0.2 \
                --layer_scale_init_value 0.0 \
                --disable_rel_pos_bias \
                --abs_pos_emb \
                --imagenet_default_mean_and_std \
                --output_dir ./evaluation/output_finetune/ \
                --data_path /earth-nas/datasets/imagenet-1k/ \
                --use_mean_pooling
                

# COCO detection and seg
python3 -m torch.distributed.launch --nproc_per_node=8 \
                ./evaluation/object_detection/train.py \
                ./evaluation/object_detection/configs/cascade_rcnn/vit_small_giou_4conv1f_coco_1x.py \
                --launcher pytorch \
                --work-dir ./evaluation/output_detection/ \
                --deterministic \
                --cfg-options model.backbone.use_checkpoint=True \
                model.pretrained=/mnt/workspace/workgroup/shaofeng.zhang/checkpoint/pospot_small_300eps.pth

# COCO detection and seg using pos_base
python3 -m torch.distributed.launch --nproc_per_node=8 \
                ./evaluation/object_detection/train.py \
                ./evaluation/object_detection/configs/cascade_rcnn/vit_base_giou_4conv1f_coco_3x.py \
                --launcher pytorch \
                --work-dir ./evaluation/output_detection/ \
                --deterministic \
                --cfg-options model.backbone.use_checkpoint=True \
                model.pretrained=/mnt/workspace/workgroup/shaofeng.zhang/checkpoint/posbot_base.pth

#  pip install mmcv-full==1.3.0 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.7.0/index.html
#  cd workgroup/shaofeng.zhang/Swin-Transformer-Ojbect-Detection
#  pip install -r requirements.txt
#  pip install -v -e .


# ade finetune
python3 -m torch.distributed.launch --nproc_per_node=8 \
                --master_port=31761 \
                ./evaluation/semantic_segmentation/train.py \
                ./evaluation/semantic_segmentation/configs/upernet/vit_base_512_ade20k_160k.py \
                --launcher pytorch \
                --work-dir ./evaluation/output_seg/ \
                --deterministic \
                --options model.backbone.use_checkpoint=True \
                model.pretrained=../checkpoint/posbot_base.pth

python3 -m torch.distributed.launch --nproc_per_node=8 \
                --master_port=53562 \
                ./evaluation/semantic_segmentation/test.py \
                ./evaluation/semantic_segmentation/configs/upernet/vit_base_512_ade20k_160k.py \
                ./evaluation/output_seg/iter_160000.pth \
                --launcher pytorch \
                --eval mIoU \
                --options model.backbone.use_checkpoint=True


