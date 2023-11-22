export CUDA_VISIBLE_DEVICES=3

python3 train.py \
    --net resnet101 \
    --gpu \
    --batchsize 16 \
    --warm 5 \
    --lr 0.001 \
    --use_softlabel \
    --use_epsilon 0.006

# optimizer
# --use_epsilon
# --use_mixup \
