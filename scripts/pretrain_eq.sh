dataset=${DMRPATH}/math23k

mkdir -p ${DMRPATH}/ckpt_pre_math23k/eq

python3 pretrain_eq.py \
        --eq_src_vocab ${dataset}/eq_src.vocab \
        --wd_src_vocab ${dataset}/wd_src.vocab \
        --tgt_vocab ${dataset}/tgt.vocab \
        --tgt_processed_vocab ${dataset}/tgt.processed.vocab \
        --train_data ${dataset}/train.txt \
        --dev_data ${dataset}/test.txt \
        --per_gpu_train_batch_size 256 \
        --dev_batch_size 256 \
        --total_train_steps 20000 \
        --ckpt ${DMRPATH}/ckpt_pre_math23k/eq \
        --gpus 1 \
        --world_size 1

# then copy the best ckpt in eq to /ckpt_pre_math23k
