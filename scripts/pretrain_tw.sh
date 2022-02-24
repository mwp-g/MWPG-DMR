dataset=${MTPATH}/math23k

mkdir -p ${MTPATH}/ckpt_pre_math23k/tw

python3 pretrain_tw.py \
        --eq_src_vocab ${dataset}/eq_src.vocab \
        --wd_src_vocab ${dataset}/wd_src.vocab \
        --tgt_vocab ${dataset}/tgt.vocab \
        --tgt_processed_vocab ${dataset}/tgt.processed.vocab \
        --train_data ${dataset}/train.txt \
        --dev_data ${dataset}/dev.txt \
        --per_gpu_train_batch_size 512 \
        --dev_batch_size 512 \
        --total_train_steps 20000 \
        --ckpt ${MTPATH}/ckpt_pre_math23k/tw \
        --gpus 1 \
        --world_size 1

# then copy the best ckpt in tw to /ckpt_pre_math23k