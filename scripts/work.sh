gen_ckpt=${MTPATH}/ckpt_gen_math23k/best.pt
pre_ckpt=${MTPATH}/ckpt_pre_math23k
dataset=${MTPATH}/math23k

mkdir -p ${MTPATH}/result

python3 work.py \
        --load_path ${gen_ckpt} \
        --index_path ${pre_ckpt} \
        --test_data ${dataset}/test.txt \
        --output_path ${MTPATH}/result \
        --eq_src_vocab_path ${dataset}/eq_src.vocab \
        --wd_src_vocab_path ${dataset}/wd_src.vocab \
        --tgt_vocab_path ${dataset}/tgt.vocab \
        --tgt_processed_vocab_path ${dataset}/tgt.processed.vocab \
        --comp_bleu


