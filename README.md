## Code of ICLR paper [ADCLR](https://openreview.net/pdf?id=10R_bcjFwJ) and ICML paper [PQCL](https://sherrylone.github.io/assets/ICML23_PQCL.pdf)

![](https://github.com/Sherrylone/sherrylone.github.io/blob/main/images/ICML23_PQCL.png)

The two papers propose query-based contrastive learning. For ADCLR, we use query crop with pixel information to learn spatial-sensitive information. For PQCL, we further mask the pixel information of the query crops, and add the relative positional embeddings to reconstruct pixel informations. Under the same setting, PQCL can get higher accuracy than ADCLR. 

To pretrain the model, run:
```
python3 -m torch.distributed.launch --nproc_per_node=8 \
        --node_rank 0 \
        --master_addr=${MASTER_ADDR:-127.0.0.1} \
        --master_port=${MASTER_PORT:-46123} \
        ./main_pos_bot.py \
        --arch pos_small \
        --output_dir ./output_100eps_005/ \
        --data_path /data/imagenet-1k/train/ \
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
```
You can change the vit_small to vit_base to learn ViT-B.

After getting the pretrained model, for linear probing, run:
```
python3 -m torch.distributed.launch --nproc_per_node=8 \
        --master_port=${MASTER_PORT:-36124} \
        ./evaluation/eval_linear.py \
            --pretrained_weights ./output_400eps_base/checkpoint.pth --n_last_blocks 4 \
            --avgpool_patchtokens 0     --arch pos_base     --checkpoint_key teacher \
            --output_dir ./evaluation/output_cls/ \
            --dist_url tcp://localhost:23142 \
            --data_path /data/imagenet-1k/
```

For detection and segmentation on COCO, run:
```
python3 -m torch.distributed.launch --nproc_per_node=8 \
                ./evaluation/object_detection/train.py \
                ./evaluation/object_detection/configs/cascade_rcnn/vit_small_giou_4conv1f_coco_1x.py \
                --launcher pytorch \
                --work-dir ./evaluation/output_detection/ \
                --deterministic \
                --cfg-options model.backbone.use_checkpoint=True \
                model.pretrained=/mnt/workspace/workgroup/shaofeng.zhang/checkpoint/pqcl_small_300eps.pth
```

