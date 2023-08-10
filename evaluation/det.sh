# COCO object detection
python3 -m torch.distributed.launch --nproc_per_node=8 \
                ./evaluation/object_detection/train.py \
                ./evaluation/object_detection/configs/cascade_rcnn/vit_small_giou_4conv1f_coco_1x.py \
                --launcher pytorch \
                --work-dir ./evaluation/output_detection/ \
                --deterministic \
                --cfg-options model.backbone.use_checkpoint=True \
                model.pretrained=/mnt/workspace/workgroup/shaofeng.zhang/pqcl/backbone/pqcl_vit_s.pth

python3 -m torch.distributed.launch --nproc_per_node=8 \
                ./evaluation/object_detection/train.py \
                ./evaluation/object_detection/configs/cascade_rcnn/vit_small_giou_4conv1f_coco_1x.py \
                --launcher pytorch \
                --work-dir ./evaluation/output_detection/ \
                --deterministic \
                --cfg-options model.backbone.use_checkpoint=True \
                model.pretrained=/mnt/workspace/workgroup/shaofeng.zhang/checkpoint/pqdistill_small_400eps_dino.pth
