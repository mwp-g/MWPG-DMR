gen_ckpt=${DMRPATH}/ckpt_gen_math23k/best.pt
pre_ckpt=${DMRPATH}/ckpt_pre_math23k
dataset=${DMRPATH}/math23k

mkdir -p ${DMRPATH}/result

python3 work.py \
        --load_path ${gen_ckpt} \
        --index_path ${pre_ckpt} \
        --test_data ${dataset}/test.txt \
        --output_path ${DMRPATH}/result \
        --eq_src_vocab_path ${dataset}/eq_src.vocab \
        --wd_src_vocab_path ${dataset}/wd_src.vocab \
        --tgt_vocab_path ${dataset}/tgt.vocab \
        --tgt_processed_vocab_path ${dataset}/tgt.processed.vocab \
        --comp_bleu


