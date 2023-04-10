dataset=${DMRPATH}/math23k \

mkdir -p ${DMRPATH}/ckpt_gen_math23k

python3 train.py \
        --eq_src_vocab ${dataset}/eq_src.vocab \
        --wd_src_vocab ${dataset}/wd_src.vocab \
        --tgt_vocab ${dataset}/tgt.vocab \
        --tgt_processed_vocab ${dataset}/tgt.processed.vocab \
        --datasets math23k \
        --retriever ${DMRPATH}/ckpt_pre_math23k \
        --train_data ${dataset}/train.txt \
        --dev_data ${dataset}/test.txt \
        --test_data ${dataset}/test.txt \
        --ckpt ${DMRPATH}/ckpt_gen_math23k \
        --MASTER_PORT 55555 \
        --topk 3 \
        --total_train_steps 100000 \
        --per_gpu_train_batch_size 512 \
        --dev_batch_size 512 \
        --gpus 2 \
        --world_size 2
