main=${MTPATH}/data
out=${MTPATH}/math23k

mkdir -p ${out}

python3 prepare/prepare.py \
--train_eq_src ${main}/math23k/train/train_eq_src.txt \
--train_wd_src ${main}/math23k/train/train_wd_src.txt \
--train_wd_orig ${main}/math23k/train/train_wd_src_orig.txt \
--train_data_tgt ${main}/math23k/train/train_tgt.txt \
--train_processed_data_tgt ${main}/math23k/train/train_processed_tgt.txt \
--eq_vocab_src ${out}/eq_src.vocab \
--wd_vocab_src ${out}/wd_src.vocab \
--vocab_tgt ${out}/tgt.vocab \
--vocab_processed_tgt ${out}/tgt.processed.vocab \
--output_file ${out}/train.txt

python3 prepare/prepare_dev_test.py \
--train_eq_src ${main}/math23k/dev/dev_eq_src.txt \
--train_wd_src ${main}/math23k/dev/dev_wd_src.txt \
--train_wd_orig ${main}/math23k/dev/dev_wd_src_orig.txt \
--train_data_tgt ${main}/math23k/dev/dev_tgt.txt \
--train_processed_data_tgt ${main}/math23k/dev/dev_processed_tgt.txt \
--output_file ${out}/dev.txt

python3 prepare/prepare_dev_test.py \
--train_eq_src ${main}/math23k/test/test_eq_src.txt \
--train_wd_src ${main}/math23k/test/test_wd_src.txt \
--train_wd_orig ${main}/math23k/test/test_wd_src_orig.txt \
--train_data_tgt ${main}/math23k/test/test_tgt.txt \
--train_processed_data_tgt ${main}/math23k/test/test_processed_tgt.txt \
--output_file ${out}/test.txt

cp ${main}/math23k/train/train.tgt.txt ${out}/train.tgt.txt
cp ${main}/math23k/train/train.processed.tgt.txt ${out}/train.processed.tgt.txt
