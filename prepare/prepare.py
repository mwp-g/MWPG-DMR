from collections import Counter
import json, re

def make_vocab(batch_seq, char_level=False):
    cnt = Counter()
    for seq in batch_seq:
        cnt.update(seq)
    if not char_level:
        return cnt
    char_cnt = Counter()
    for x, y in cnt.most_common():
        for ch in list(x):
            char_cnt[ch] += y
    return cnt, char_cnt

def write_vocab(vocab, path):
    with open(path, 'w') as fo:
        for x, y in vocab.most_common():
            fo.write('%s\t%d\n'%(x,y))

import argparse
def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_eq_src', type=str)
    parser.add_argument('--train_wd_src', type=str)
    parser.add_argument('--train_wd_orig', type=str)

    parser.add_argument('--train_data_tgt', type=str)
    parser.add_argument('--train_processed_data_tgt', type=str)

    parser.add_argument('--output_file', type=str)

    parser.add_argument('--eq_vocab_src', type=str)
    parser.add_argument('--wd_vocab_src', type=str)
    parser.add_argument('--vocab_tgt', type=str)
    parser.add_argument('--vocab_processed_tgt', type=str)

    parser.add_argument('--ratio', type=float, default=1.5)
    parser.add_argument('--min_len', type=int, default=1)
    parser.add_argument('--max_len', type=str, default=250)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_config()

    print('make vocabularies')
    fo = open(args.output_file, 'w')
    eq_src_lines = []
    wd_src_lines = []
    tgt_processed_lines = []
    tgt_lines = []
    tot_lines = 0
    for eq_src_line, wd_src_line, tgt_line, wd_orig_line, tgt_processed_line in zip(open(args.train_eq_src).readlines(),
                                                                                    open(args.train_wd_src).readlines(),
                                                                                    open(args.train_data_tgt).readlines(),
                                                                                    open(args.train_wd_orig).readlines(),
                                                                                    open(args.train_processed_data_tgt).readlines()):
        eq_src_line = eq_src_line.strip().split()
        wd_src_line = wd_src_line.strip().split()
        wd_orig_line = wd_orig_line.strip().split()
        tgt_line = tgt_line.strip().split()
        tgt_processed_line = tgt_processed_line.strip().split()
        tot_lines += 1
        # if args.min_len <= len(src_line) <= args.max_len and args.min_len <= len(tgt_line) <= args.max_len:
        #     if len(src_line)/len(tgt_line) > args.ratio:
        #         continue
        #     if len(tgt_line)/len(src_line) > args.ratio:
        #         continue
            # fo.write(' '.join(src_line) + '\t' + ' '.join(tgt_line) + '\n')
        fo.write(' '.join(eq_src_line) + '\t' + ' '.join(wd_src_line) + '\t' + ' '.join(tgt_line) + '\t' + ' '.join(tgt_processed_line) + '\t' + ' '.join(wd_orig_line) + '\n')
        # fo.write(' '.join(eq_src_line) + '\t' + ' '.join(wd_src_line) + '\t' + ' '.join(tgt_line) + '\n' )

        eq_src_lines.append(eq_src_line)
        wd_src_lines.append(wd_src_line)
        tgt_lines.append(tgt_line)
        # tgt_lines.append(tgt_processed_line)
        tgt_processed_lines.append(tgt_processed_line)
        # tgt_processed_lines.append(tgt_line)
    fo.close()
    eq_src_vocab = make_vocab(eq_src_lines)
    wd_src_vocab = make_vocab(wd_src_lines)
    tgt_vocab = make_vocab(tgt_lines)
    tgt_processed_vocab = make_vocab(tgt_processed_lines)

    print(args.output_file, len(tgt_lines), tot_lines)

    print('write vocabularies')
    write_vocab(eq_src_vocab, args.eq_vocab_src)
    write_vocab(wd_src_vocab, args.wd_vocab_src)
    write_vocab(tgt_vocab, args.vocab_tgt)
    write_vocab(tgt_processed_vocab, args.vocab_processed_tgt)

