import torch
import argparse, os, logging
import random
import torch.multiprocessing as mp
import torch.distributed as dist
import numpy as np
import math
import sys
sys.path.append('../MWPG-DMR')
from utils.data import Vocab, BOS, EOS, UNK, ListsToTensor, _back_to_txt_for_check
from utils.optim import Adam, get_linear_schedule_with_warmup
from utils.utils import move_to_device, set_seed, average_gradients, Statistics
from modules.retriever import MatchingModel
from collections import Counter
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
logger = logging.getLogger(__name__)

def parse_config():
    parser = argparse.ArgumentParser()
    # vocabs
    parser.add_argument('--eq_src_vocab', type=str)
    parser.add_argument('--wd_src_vocab', type=str)
    parser.add_argument('--tgt_vocab', type=str)
    parser.add_argument('--tgt_processed_vocab', type=str)

    # architecture
    parser.add_argument('--embed_dim', type=int, default=512)
    parser.add_argument('--ff_embed_dim', type=int, default=2048)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--output_dim', type=int, default=256)

    # dropout / label_smoothing
    parser.add_argument('--worddrop', type=float, default=0.1)
    # if worddrop < 0, we are using idf-based masking
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--label_smoothing', type=float, default=0.1)

    # training
    parser.add_argument('--bow', action='store_true', default=False)
    parser.add_argument('--resume_ckpt', type=str, default=None)
    parser.add_argument('--additional_negs', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--total_train_steps', type=int, default=15000)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--per_gpu_train_batch_size', type=int, default=256)
    parser.add_argument('--dev_batch_size', type=int, default=256)

    # IO
    parser.add_argument('--train_data', type=str)
    # /data/wangxiaowei/copyisallyouneed/math23k/train.txt
    parser.add_argument('--dev_data', type=str)
    # /data/wangxiaowei/copyisallyouneed/math23k/dev.txt
    parser.add_argument('--ckpt', type=str)
    # /data/wangxiaowei/copyisallyouneed/ckpt.exp.pretrain4
    parser.add_argument('--print_every', type=int, default=500)
    parser.add_argument('--eval_every', type=int, default=500)

    # distributed training
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--MASTER_ADDR', type=str, default='localhost')
    parser.add_argument('--MASTER_PORT', type=str, default='55555')
    parser.add_argument('--start_rank', type=int, default=0)
    return parser.parse_args()

def compute_idf(sents):
    df = Counter()
    n = 0
    for sent in sents:
        for word in set(sent):
            df[word] += 1
        n += 1

    idf = dict()
    for word in df:
        idf[word] = 1 + np.log((1+n) / (1 + df[word]))
    idf[BOS] = 123456789
    return idf

def idf_based_mask(sents, idf):
    # 1/3 * 0 + 2/3 * 1/2 = 1/3
    ret = []
    for sent in sents:
        indices = list(range(len(sent)))
        lowest = math.floor(len(sent) * 2 / 3)
        masked_sent = [ w for w in sent]
        for i in sorted(indices, key=lambda x:idf[sent[x]])[:lowest]:
            masked_sent[i] = sent[i] if random.random() < 0.5 else UNK
        ret.append(masked_sent)
    return ret

class DataLoader(object):
    def __init__(self, vocabs, filename, batch_size, worddrop=0., max_seq_len=256, addition=True):
        self.vocabs = vocabs
        self.batch_size = batch_size
        self.worddrop = worddrop
        self.addition = addition

        # src_tokens, tgt_tokens = [], []
        eq_tokens, wd_tokens, tgt_tokens, tgt_processed_tokens = [], [], [], []
        adt_sents = []
        for line in open(filename).readlines():
            if self.addition:
                eq, wd, tgt, tgt_processed, wd_orig, *adt = line.strip().split('\t')
                adt = [ x.split()[:max_seq_len] for x in adt[::2] ]
                adt_sents.append(adt)
            else:
                eq, wd, tgt, tgt_processed, _ = line.strip().split('\t')
            eq, wd, tgt, tgt_processed = eq.split()[:max_seq_len], wd.split()[:max_seq_len], tgt.split()[:max_seq_len], tgt_processed.split()[:max_seq_len]
            eq_tokens.append(eq)
            wd_tokens.append(wd)
            tgt_tokens.append(tgt)
            tgt_processed_tokens.append(tgt_processed)

        # self.src = src_tokens
        self.eq = eq_tokens
        self.wd = wd_tokens
        self.tgt = tgt_tokens
        self.tgt_processed = tgt_processed_tokens
        self.adt = adt_sents
        # self.idf_src = compute_idf(self.src)
        self.idf_eq = compute_idf(self.eq)
        self.idf_wd = compute_idf(self.wd)
        self.idf_tgt = compute_idf(self.tgt)
        self.idf_tgt_processed = compute_idf(self.tgt_processed)

    def batchify(self, data):

        eq_tokens = [[BOS] + x['eq_tokens'] for x in data]
        wd_tokens = [[BOS] + x['wd_tokens'] for x in data]
        tgt_tokens = [[BOS] + x['tgt_tokens'] for x in data]
        tgt_processed_tokens = [[BOS] + x['tgt_processed_tokens'] for x in data]

        if 'adt_tokens' in data[0]:
            adt_tokens = [[BOS] + x['adt_tokens'] for x in data]
            tgt_tokens = tgt_tokens + adt_tokens
            tgt_processed_tokens = tgt_processed_tokens + adt_tokens

        #ori_src_tokens = ListsToTensor(src_tokens, self.vocabs['src'])
        #ori_tgt_tokens = ListsToTensor(tgt_tokens, self.vocabs['tgt'])
        
        # if self.worddrop < 0.:
        #     src_tokens = ListsToTensor(idf_based_mask(src_tokens, self.idf_src), self.vocabs['src'])
        #     tgt_tokens = ListsToTensor(idf_based_mask(tgt_tokens, self.idf_tgt), self.vocabs['tgt'])
        # else:
        eq_tokens = ListsToTensor(eq_tokens, self.vocabs['eq_src'], self.worddrop)
        wd_tokens = ListsToTensor(wd_tokens, self.vocabs['wd_src'], self.worddrop)
        tgt_tokens = ListsToTensor(tgt_tokens, self.vocabs['tgt'], self.worddrop)
        tgt_processed_tokens = ListsToTensor(tgt_processed_tokens, self.vocabs['tgt_processed'], self.worddrop)
        ret = {
            'eq_tokens': eq_tokens,
            'wd_tokens': wd_tokens,
            'tgt_tokens': tgt_tokens,
            'tgt_processed_tokens': tgt_processed_tokens,
        }
        return ret

    def __len__(self):
        return len(self.eq)

    def __iter__(self):
        indices = np.random.permutation(len(self))
        #indices = np.arange(len(self))
        
        cur = 0
        while cur < len(indices):
            # if self.addition:
            #     data = [{'src_tokens':self.src[i], 'tgt_tokens':self.tgt[i], 'adt_tokens':random.choice(self.adt[i])} for i in indices[cur:cur+self.batch_size]]
            # else:
            data = [{'eq_tokens':self.eq[i], 'wd_tokens':self.wd[i], 'tgt_tokens':self.tgt[i], 'tgt_processed_tokens':self.tgt_processed[i]}for i in indices[cur:cur + self.batch_size]]
            cur += self.batch_size
            yield self.batchify(data)

@torch.no_grad()
def validate(model, dev_data, device):
    model.eval()
    q_list = []
    r_list = []
    for batch in dev_data:
        batch = move_to_device(batch, device)
        q = model.query_encoder(batch['eq_tokens'], batch['wd_tokens'])
        r = model.response_encoder(batch['tgt_processed_tokens'])
        q_list.append(q)
        r_list.append(r)
    q = torch.cat(q_list, dim=0)
    r = torch.cat(r_list, dim=0)

    bsz = q.size(0)
    scores = torch.mm(q, r.t())  # bsz x bsz
    gold = torch.arange(bsz, device=scores.device)
    _, pred = torch.max(scores, -1)
    acc = torch.sum(torch.eq(gold, pred).float()) / bsz
    return acc

def main(args, local_rank):

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

    vocabs = dict()
    vocabs['eq_src'] = Vocab(args.eq_src_vocab, 0, [BOS, EOS])
    vocabs['wd_src'] = Vocab(args.wd_src_vocab, 0, [BOS, EOS])
    vocabs['tgt'] = Vocab(args.tgt_vocab, 0, [BOS, EOS])
    vocabs['tgt_processed'] = Vocab(args.tgt_processed_vocab, 0, [BOS, EOS])

    if args.world_size == 1 or (dist.get_rank() == 0):
        logger.info(args)
        for name in vocabs:
            logger.info("vocab %s, size %d, coverage %.3f", name, vocabs[name].size, vocabs[name].coverage)

    set_seed(19940117)

    #device = torch.device('cpu')


    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.resume_ckpt:
        model = MatchingModel.from_pretrained(vocabs, args.resume_ckpt)
    else:
        model = MatchingModel.from_params(vocabs, args.layers, args.embed_dim, args.ff_embed_dim, args.num_heads, args.dropout, args.output_dim, args.bow)

    if args.world_size > 1:
        set_seed(19940117 + dist.get_rank())

    # model = torch.nn.DataParallel(model, device_ids=[0, 1, 2])

    model = model.to(device)

    if args.resume_ckpt:
        dev_data = DataLoader(vocabs, args.dev_data, args.dev_batch_size, addition=args.additional_negs)
        acc = validate(model, dev_data, device)
        logger.info("initialize from %s, initial acc %.2f", args.resume_ckpt, acc)

    optimizer = Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    lr_schedule = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, args.total_train_steps)
    train_data = DataLoader(vocabs, args.train_data, args.per_gpu_train_batch_size, worddrop=args.worddrop, addition=args.additional_negs)
    global_step, step, epoch = 0, 0, 0
    tr_stat = Statistics()
    logger.info("start training")
    model.train()
    while global_step <= args.total_train_steps:
        for batch in train_data:
            batch = move_to_device(batch, device)
            loss, acc, bsz = model(batch['eq_tokens'], batch['wd_tokens'], batch['tgt_processed_tokens'], args.label_smoothing)
            tr_stat.update({'loss':loss.item() * bsz,
                            'nsamples': bsz,
                            'acc':acc * bsz})
            tr_stat.step()
            loss.backward()

            step += 1
            if not (step % args.gradient_accumulation_steps == -1 % args.gradient_accumulation_steps):
                continue

            if args.world_size > 1:
                average_gradients(model)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_schedule.step()
            optimizer.zero_grad()
            global_step += 1

            if args.world_size == 1 or (dist.get_rank() == 0):
                if global_step % args.print_every == -1 % args.print_every:
                    logger.info("epoch %d, step %d, loss %.3f, acc %.3f", epoch, global_step, tr_stat['loss']/tr_stat['nsamples'], tr_stat['acc']/tr_stat['nsamples'])
                    tr_stat = Statistics()
                if global_step > args.warmup_steps and global_step % args.eval_every == -1 % args.eval_every:
                    dev_data = DataLoader(vocabs, args.dev_data, args.dev_batch_size, addition=args.additional_negs)
                    acc = validate(model, dev_data, device)
                    logger.info("epoch %d, step %d, dev, dev acc %.2f", epoch, global_step, acc)
                    save_path = '%s/epoch%d_batch%d_acc%.2f'%(args.ckpt, epoch, global_step, acc)
                    model.save(args, save_path)
                    model.train()
            if global_step > args.total_train_steps:
                break
        epoch += 1
    logger.info('rank %d, finish training after %d steps', local_rank, global_step)

def init_processes(local_rank, args, backend='nccl'):
    os.environ['MASTER_ADDR'] = args.MASTER_ADDR
    os.environ['MASTER_PORT'] = args.MASTER_PORT
    dist.init_process_group(backend, rank=args.start_rank+local_rank, world_size=args.world_size)
    main(args, local_rank)

if __name__ == "__main__":
    args = parse_config()
    if not os.path.exists(args.ckpt):
        os.mkdir(args.ckpt)

    if args.world_size == 1:
        main(args, 0)
        exit(0)

    mp.spawn(init_processes, args=(args,), nprocs=args.gpus)
