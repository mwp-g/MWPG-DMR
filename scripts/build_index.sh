dataset=${MTPATH}/math23k

ckpt_folder=${MTPATH}/ckpt_pre_math23k

# the input_file contains target-side sentences, and pls make sure to remove duplicates

python3 build_index_eq.py \
        --input_file ${dataset}/train.processed.tgt.txt \
        --args_path ${ckpt_folder}/eq_args \
        --ckpt_path ${ckpt_folder}/eq_response_encoder \
        --vocab_path ${dataset}/tgt.processed.vocab \
        --index_path ${ckpt_folder}/eq_mips_index \

python3 build_index_eq.py \
        --input_file ${dataset}/train.processed.tgt.txt \
        --args_path ${ckpt_folder}/eq_args \
        --ckpt_path ${ckpt_folder}/eq_response_encoder \
        --vocab_path ${dataset}/tgt.processed.vocab \
        --index_path ${ckpt_folder}/eq_mips_index \
        --only_dump_feat

python3 build_index_tw.py \
        --input_file ${dataset}/train.tgt.txt \
        --args_path ${ckpt_folder}/wd_args \
        --ckpt_path ${ckpt_folder}/wd_response_encoder \
        --vocab_path ${dataset}/tgt.vocab \
        --index_path ${ckpt_folder}/wd_mips_index \

python3 build_index_tw.py \
        --input_file ${dataset}/train.tgt.txt \
        --args_path ${ckpt_folder}/wd_args \
        --ckpt_path ${ckpt_folder}/wd_response_encoder \
        --vocab_path ${dataset}/tgt.vocab \
        --index_path ${ckpt_folder}/wd_mips_index \
        --only_dump_feat

cp ${dataset}/train/train.tgt.txt ${ckpt_folder}/candidates.txt
cp ${dataset}/train/train.processed.tgt.txt ${ckpt_folder}/candidates.processed.txt
