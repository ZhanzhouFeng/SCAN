accelerate launch --mixed_precision=bf16 --num_processes 2 --multi_gpu \
    scripts/train_c2i.py  --parallel --exp-name test --log-every 20 --visualize-every 500 --ckpt-every 10000 --keep-last-k 2 \
    --dataset imagenetraw --data-path /datacube_nas/datasets/labels/dingsunbao/qiyin.hqy/imagenet/train \
    --data-anno /new_shanghai/fengzheng.fzz/data/data_configs/ImageNet_Train_paths_mainStation.txt \
    --imagenet-class /datacube_nas/datasets/labels/dingsunbao/qiyin.hqy/imagenet/labels.txt \
    --config configs/SCAN/SCANmaskar_l_0.3b_llamagen_dis.yaml  --num-workers 8 --global_batch_size 128 --random_ratio 0.1