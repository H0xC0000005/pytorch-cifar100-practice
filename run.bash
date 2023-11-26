export CUDA_VISIBLE_DEVICES=0

python3 train.py \
    --net resnet101 \
    --gpu \
    --batchsize 16 \
    --warm 5 \
    --lr 0.001 \
    --use_softlabel \
    --use_epsilon 0.003

# optimizer
# --use_epsilon
# --use_mixup \
