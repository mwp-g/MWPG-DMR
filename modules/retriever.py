import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import math
import os, time, random, logging

from modules.transformer import Transformer, SinusoidalPositionalEmbedding, Embedding
from utils.utils import move_to_device, asynchronous_load
from modules.module import label_smoothed_nll_loss, layer_norm, MonoEncoder, RnnEncoder
from utils.mips import MIPS, augment_query, augment_data, l2_to_ip
from utils.data import BOS, EOS, ListsToTensor, _back_to_txt_for_check

logger = logging.getLogger(__name__)
class Retriever(nn.Module):
    def __init__(self, vocabs, eq_model, wd_model, eq_mips, wd_mips, eq_mips_max_norm, wd_mips_max_norm, mem_pool, mem_processed_pool, eq_mem_feat_or_feat_maker, wd_mem_feat_or_feat_maker, num_heads, topk, gpuid):
        super(Retriever, self).__init__()
        self.eq_model = eq_model
        self.wd_model = wd_model
        self.mem_pool = mem_pool
        self.mem_processed_pool = mem_processed_pool
        self.eq_mem_feat_or_feat_maker = eq_mem_feat_or_feat_maker
        self.wd_mem_feat_or_feat_maker = wd_mem_feat_or_feat_maker
        self.num_heads = num_heads
        self.topk = topk
        self.vocabs = vocabs
        self.gpuid = gpuid
        self.eq_mips = eq_mips
        self.wd_mips = wd_mips
        if self.gpuid >= 0:
            self.eq_mips.to_gpu(gpuid=self.gpuid)
            self.wd_mips.to_gpu(gpuid=self.gpuid)
        self.eq_mips_max_norm = eq_mips_max_norm
        self.wd_mips_max_norm = wd_mips_max_norm

    @classmethod
    def from_pretrained(cls, num_heads, vocabs, input_dir, nprobe, topk, gpuid, use_response_encoder=False):
        eq_model_args = torch.load(os.path.join(input_dir, 'eq_args'))
        wd_model_args = torch.load(os.path.join(input_dir, 'wd_args'))

        eq_model = MultiProjEncoder_spc.from_pretrained_projencoder(num_heads, vocabs['eq_src'], vocabs['wd_src'], eq_model_args, os.path.join(input_dir, 'eq_query_encoder'))
        wd_model = MultiProjEncoder_wd.from_pretrained_projencoder(num_heads, vocabs['eq_src'], vocabs['wd_src'], wd_model_args, os.path.join(input_dir, 'wd_query_encoder'))

        mem_pool = [line.strip().split() for line in open(os.path.join(input_dir, 'candidates.txt')).readlines()]
        mem_processed_pool = [line.strip().split() for line in open(os.path.join(input_dir, 'candidates.processed.txt')).readlines()]
        if use_response_encoder:
            eq_mem_feat_or_feat_maker = ProjEncoder.from_pretrained(vocabs['tgt_processed'], eq_model_args, os.path.join(input_dir, 'eq_response_encoder'))
            wd_mem_feat_or_feat_maker = ProjEncoder.from_pretrained(vocabs['tgt'], wd_model_args, os.path.join(input_dir, 'wd_response_encoder'))
        else:
            eq_mem_feat_or_feat_maker = torch.load(os.path.join(input_dir, 'eq_feat.pt'))
            wd_mem_feat_or_feat_maker = torch.load(os.path.join(input_dir, 'wd_feat.pt'))
        eq_mips = MIPS.from_built(os.path.join(input_dir, 'eq_mips_index'), nprobe=nprobe)
        wd_mips = MIPS.from_built(os.path.join(input_dir, 'wd_mips_index'), nprobe=nprobe)
        eq_mips_max_norm = torch.load(os.path.join(input_dir, 'eq_max_norm.pt'))
        wd_mips_max_norm = torch.load(os.path.join(input_dir, 'wd_max_norm.pt'))
        retriever = cls(vocabs, eq_model, wd_model, eq_mips, wd_mips, eq_mips_max_norm, wd_mips_max_norm, mem_pool, mem_processed_pool, eq_mem_feat_or_feat_maker, wd_mem_feat_or_feat_maker, num_heads, topk, gpuid)
        return retriever

    def drop_index(self):
        self.eq_mips.reset()
        self.eq_mips = None
        self.eq_mips_max_norm = None
        self.wd_mips.reset()
        self.wd_mips = None
        self.wd_mips_max_norm = None

    def update_index(self, index_dir, nprobe):
        self.eq_mips = MIPS.from_built(os.path.join(index_dir, 'eq_mips_index'), nprobe=nprobe)
        self.wd_mips = MIPS.from_built(os.path.join(index_dir, 'wd_mips_index'), nprobe=nprobe)
        if self.gpuid >= 0:
            self.eq_mips.to_gpu(gpuid=self.gpuid)
            self.wd_mips.to_gpu(gpuid=self.gpuid)
        self.eq_mips_max_norm = torch.load(os.path.join(index_dir, 'eq_max_norm.pt'))
        self.wd_mips_max_norm = torch.load(os.path.join(index_dir, 'wd_max_norm.pt'))

    def rebuild_index(self, index_dir, batch_size=2048, add_every=1000000, index_type='IVF1024_HNSW32,SQ8', norm_th=999, max_training_instances=1000000, max_norm_cf=1.0, nprobe=64, efSearch=128):
        if not os.path.exists(index_dir):
            os.mkdir(index_dir)
        eq_max_norm = None
        wd_max_norm = None
        data_eq = [[' '.join(x), i] for i, x in enumerate(self.mem_processed_pool)]
        data_wd = [[' '.join(x), i] for i, x in enumerate(self.mem_pool)]
        random.shuffle(data_eq)
        random.shuffle(data_wd)
        used_data_eq = [x[0] for x in data_eq[:max_training_instances]]
        used_ids_eq = np.array([x[1] for x in data_eq[:max_training_instances]])
        used_data_wd = [x[0] for x in data_wd[:max_training_instances]]
        used_ids_wd = np.array([x[1] for x in data_wd[:max_training_instances]])
        logger.info('Computing feature for training')
        eq_used_data, eq_used_ids, eq_max_norm = get_features(batch_size, norm_th, self.vocabs['tgt_processed'], self.eq_mem_feat_or_feat_maker, used_data_eq, used_ids_eq, max_norm_cf=max_norm_cf)
        wd_used_data, wd_used_ids, wd_max_norm = get_features(batch_size, norm_th, self.vocabs['tgt'], self.wd_mem_feat_or_feat_maker, used_data_wd, used_ids_wd, max_norm_cf=max_norm_cf)
        torch.cuda.empty_cache()
        logger.info('Using eq %d instances for training', eq_used_data.shape[0])
        logger.info('Using eq %d instances for training', wd_used_data.shape[0])
        eq_mips = MIPS(self.eq_model.output_dim+1, index_type, efSearch=efSearch, nprobe=nprobe)
        eq_mips.to_gpu()
        eq_mips.train(eq_used_data)
        eq_mips.to_cpu()
        eq_mips.add_with_ids(eq_used_data, eq_used_ids)
        data_eq = data_eq[max_training_instances:]

        wd_mips = MIPS(self.wd_model.output_dim + 1, index_type, efSearch=efSearch, nprobe=nprobe)
        wd_mips.to_gpu()
        wd_mips.train(wd_used_data)
        wd_mips.to_cpu()
        wd_mips.add_with_ids(wd_used_data, wd_used_ids)
        data_wd = data_wd[max_training_instances:]

        torch.save(eq_max_norm, os.path.join(index_dir, 'eq_max_norm.pt'))
        torch.save(wd_max_norm, os.path.join(index_dir, 'wd_max_norm.pt'))
        
        cur_eq = 0
        while cur_eq < len(data_eq):
            used_data_eq = [x[0] for x in data_eq[cur_eq:cur_eq+add_every]]
            used_ids_eq = np.array([x[1] for x in data_eq[cur_eq:cur_eq+add_every]])
            cur_eq += add_every
            logger.info('Computing feature for indexing')
            eq_used_data, eq_used_ids, _ = get_features(batch_size, norm_th, self.vocabs['tgt_processed'], self.eq_mem_feat_or_feat_maker, used_data_eq, used_ids_eq, eq_max_norm)
            logger.info('Adding eq %d instances to index', eq_used_data.shape[0])
            eq_mips.add_with_ids(eq_used_data, eq_used_ids)

        cur_wd = 0
        while cur_wd < len(data_wd):
            used_data_wd = [x[0] for x in data_wd[cur_wd:cur_wd+add_every]]
            used_ids_wd = np.array([x[1] for x in data_wd[cur_wd:cur_wd+add_every]])
            cur_wd += add_every
            logger.info('Computing feature for indexing')
            wd_used_data, wd_used_ids, _ = get_features(batch_size, norm_th, self.vocabs['tgt'], self.wd_mem_feat_or_feat_maker, used_data_wd, used_ids_wd, wd_max_norm)
            logger.info('Adding eq %d instances to index', wd_used_data.shape[0])
            eq_mips.add_with_ids(wd_used_data, wd_used_ids)

        eq_mips.save(os.path.join(index_dir, 'eq_mips_index'))
        wd_mips.save(os.path.join(index_dir, 'wd_mips_index'))

    def work(self, inp, allow_hit):
        eq_tokens = inp['eq_tokens']
        wd_tokens = inp['wd_tokens']
        eq_feat, eq, eq_mask = self.eq_model(eq_tokens, eq_tokens, return_src=True)
        wd_feat, wd, wd_mask = self.wd_model(wd_tokens, wd_tokens, return_src=True)
        eq_num_heads, eq_bsz, eq_dim = eq_feat.size()
        wd_num_heads, wd_bsz, wd_dim = wd_feat.size()
        assert eq_num_heads == self.num_heads
        assert wd_num_heads == self.num_heads
        topk_eq = self.topk
        topk_wd = self.topk
        eq_vecsq = eq_feat.reshape(eq_num_heads * eq_bsz, -1).detach().cpu().numpy()
        wd_vecsq = wd_feat.reshape(wd_num_heads * wd_bsz, -1).detach().cpu().numpy()
        #retrieval_start = time.time()
        eq_vecsq = augment_query(eq_vecsq)
        wd_vecsq = augment_query(wd_vecsq)
        eq_D, eq_I = self.eq_mips.search(eq_vecsq, topk_eq + 1)
        wd_D, wd_I = self.wd_mips.search(wd_vecsq, topk_wd + 1)
        eq_D = l2_to_ip(eq_D, eq_vecsq, self.eq_mips_max_norm) / (self.eq_mips_max_norm * self.eq_mips_max_norm)
        wd_D = l2_to_ip(wd_D, wd_vecsq, self.wd_mips_max_norm) / (self.wd_mips_max_norm * self.wd_mips_max_norm)
        eq_indices = torch.zeros(topk_eq, eq_num_heads, eq_bsz, dtype=torch.long)
        wd_indices = torch.zeros(topk_wd, wd_num_heads, wd_bsz, dtype=torch.long)

        for i, (Ii, Di) in enumerate(zip(eq_I, eq_D)):
            bid, hid = i % eq_bsz, i // eq_bsz
            tmp_list = []
            for pred, _ in zip(Ii, Di):
                if allow_hit or self.mem_processed_pool[pred] != inp['tgt_raw_sents'][bid]:
                    tmp_list.append(pred)
            tmp_list = tmp_list[:topk_eq]
            assert len(tmp_list) == topk_eq
            eq_indices[:, hid, bid] = torch.tensor(tmp_list)

        for i, (Ii, Di) in enumerate(zip(wd_I, wd_D)):
            bid, hid = i % wd_bsz, i // wd_bsz
            wd_tmp_list = []
            for pred, _ in zip(Ii, Di):
                if allow_hit or self.mem_pool[pred]!=inp['tgt_raw_sents'][bid]:
                    wd_tmp_list.append(pred)
            wd_tmp_list = wd_tmp_list[:topk_wd]
            assert len(wd_tmp_list) == topk_wd
            wd_indices[:, hid, bid] = torch.tensor(wd_tmp_list)

        eq_all_mem_tokens = []
        wd_all_mem_tokens = []

        for idx in eq_indices.view(-1).tolist():
            #TODO self.mem_pool[idx] +[EOS]
            eq_all_mem_tokens.append([BOS] + self.mem_processed_pool[idx])

        for idx in wd_indices.view(-1).tolist():
            #TODO self.mem_pool[idx] +[EOS]
            wd_all_mem_tokens.append([BOS] + self.mem_pool[idx])

        eq_all_mem_tokens = ListsToTensor(eq_all_mem_tokens, self.vocabs['tgt_processed'])
        wd_all_mem_tokens = ListsToTensor(wd_all_mem_tokens, self.vocabs['tgt'])

        eq_max_mem_len = int(1.5 * eq_tokens.shape[0])
        wd_max_mem_len = int(1.5 * wd_tokens.shape[0])
        eq_all_mem_tokens = move_to_device(eq_all_mem_tokens[:eq_max_mem_len, :], inp['eq_tokens'].device)
        wd_all_mem_tokens = move_to_device(wd_all_mem_tokens[:wd_max_mem_len, :], inp['wd_tokens'].device)
       
        if torch.is_tensor(self.eq_mem_feat_or_feat_maker):
            eq_all_mem_feats = self.eq_mem_feat_or_feat_maker[eq_indices].to(eq_feat.device)
        else:
            eq_all_mem_feats = self.eq_mem_feat_or_feat_maker(eq_all_mem_tokens).view(topk_eq, eq_num_heads, eq_bsz, eq_dim)

        if torch.is_tensor(self.wd_mem_feat_or_feat_maker):
            wd_all_mem_feats = self.wd_mem_feat_or_feat_maker[wd_indices].to(wd_feat.device)
        else:
            wd_all_mem_feats = self.wd_mem_feat_or_feat_maker(wd_all_mem_tokens).view(topk_wd, wd_num_heads, wd_bsz, wd_dim)

        eq_all_mem_scores = torch.sum(eq_feat.unsqueeze(0) * eq_all_mem_feats, dim=-1) / (self.eq_mips_max_norm ** 2)
        wd_all_mem_scores = torch.sum(wd_feat.unsqueeze(0) * wd_all_mem_feats, dim=-1) / (self.wd_mips_max_norm ** 2)

        mem_ret = {}
        eq_indices = eq_indices.view(-1, eq_bsz).transpose(0, 1).tolist()
        wd_indices = wd_indices.view(-1, wd_bsz).transpose(0, 1).tolist()
        mem_ret['retrieval_raw_sents'] = [ [self.mem_processed_pool[idx] for idx in ind] for ind in eq_indices]
        mem_ret['all_mem_tokens'] = eq_all_mem_tokens
        mem_ret['all_mem_scores'] = eq_all_mem_scores
        mem_ret['wd_retrieval_raw_sents'] = [[self.mem_pool[idx] for idx in ind] for ind in wd_indices]
        mem_ret['wd_all_mem_tokens'] = wd_all_mem_tokens
        mem_ret['wd_all_mem_scores'] = wd_all_mem_scores
        return mem_ret


class MatchingModel(nn.Module):
    def __init__(self, query_encoder, response_encoder, bow=False):
        super(MatchingModel, self).__init__()
        self.query_encoder = query_encoder
        self.response_encoder = response_encoder

    def forward(self, eq_orig, wd_orig, response, label_smoothing=0.):
        ''' query and response: [seq_len, batch_size]
        '''
        eq_len, bsz = eq_orig.size()
        q, q_src, _ = self.query_encoder(eq_orig, wd_orig, return_src=True)
        r, r_src, _ = self.response_encoder(response, return_src=True)

        q_src = torch.mean(q_src, 0)
        # q_src = q_src[0, :, :]         # dolphin18k

        r_src = r_src[0, :, :]
        scores = torch.mm(q, r.t()) # bsz x (bsz + adt)
        gold = torch.arange(bsz, device=scores.device)
        _, pred = torch.max(scores, -1)
        acc = torch.sum(torch.eq(gold, pred).float()) / bsz
        log_probs = F.log_softmax(scores, -1)
        loss, _ = label_smoothed_nll_loss(log_probs, gold, label_smoothing, sum=True)
        loss = loss / bsz
        if self.bow:
            loss_bow_eq = self.eq_bow(r_src, eq_orig.transpose(0, 1))
            # loss_bow_wd = self.wd_bow(r_src, wd_orig.transpose(0, 1))
            loss_bow_r = self.response_bow(q_src, response.transpose(0, 1))
            # loss = loss + loss_bow_eq + loss_bow_r
            loss = loss + loss_bow_eq + loss_bow_r
            # loss = loss + loss_bow_eq + loss_bow_r + loss_bow_wd
        return loss, acc, bsz
    def work(self, eq_orig, wd_orig, response):
        ''' query and response: [seq_len x batch_size ]
        '''
        _, bsz = eq_orig.size()
        q = self.query_encoder(eq_orig, wd_orig)
        r = self.response_encoder(response)
        scores = torch.sum(q * r, -1)
        return scores

    def save(self, model_args, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.query_encoder.state_dict(), os.path.join(output_dir, 'eq_query_encoder'))
        torch.save(self.response_encoder.state_dict(), os.path.join(output_dir, 'eq_response_encoder'))
        torch.save(model_args, os.path.join(output_dir, 'eq_args'))

    @classmethod
    def from_params(cls, vocabs, layers, embed_dim, ff_embed_dim, num_heads, dropout, output_dim, bow):
        query_encoder = ProjEncoder_spc(vocabs['eq_src'], vocabs['wd_src'], layers, embed_dim, ff_embed_dim, num_heads, dropout, output_dim)
        response_encoder = ProjEncoder(vocabs['tgt_processed'], layers, embed_dim, ff_embed_dim, num_heads, dropout, output_dim)
        model = cls(query_encoder, response_encoder, bow)
        return model
    
    @classmethod
    def from_pretrained(cls, vocabs, input_dir):
        model_args = torch.load(os.path.join(input_dir, 'eq_args'))
        query_encoder = ProjEncoder_spc.from_pretrained(vocabs['eq_src'], vocabs['wd_src'], model_args, os.path.join(input_dir, 'eq_query_encoder'))
        response_encoder = ProjEncoder.from_pretrained(vocabs['tgt_processed'], model_args, os.path.join(input_dir, 'eq_response_encoder'))
        model = cls(query_encoder, response_encoder)
        return model

class MultiProjEncoder_spc(nn.Module):
    def __init__(self, num_proj_heads, vocab_eq, vocab_wd, layers, embed_dim, ff_embed_dim, num_heads, dropout, output_dim):
        super(MultiProjEncoder_spc, self).__init__()
        # self.eq_encoder = MonoEncoder(vocab_eq, layers, embed_dim, ff_embed_dim, num_heads, dropout)
        self.eq_encoder = RnnEncoder(vocab_eq, embed_dim, 512, 2)
        # self.wd_encoder = MonoEncoder(vocab_wd, layers, embed_dim, ff_embed_dim, num_heads, dropout)
        self.proj = nn.Linear(embed_dim, num_proj_heads*output_dim)
        self.num_proj_heads = num_proj_heads
        self.output_dim = output_dim
        self.dropout = dropout

    def forward(self, input_eq, input_wd, batch_first=False, return_src=False):
        if batch_first:
            input_eq = input_eq.t()
            # input_wd = input_wd.t()
        eq, eq_mask = self.eq_encoder(input_eq)
        # wd, wd_mask = self.wd_encoder(input_wd)
        ret = torch.mean(eq, 0)
        # ret_wd = wd[0, :, :]
        # ret = (ret_eq + ret_wd) / 2
        ret = F.dropout(ret, p=self.dropout, training=self.training)
        ret = self.proj(ret).view(-1, self.num_proj_heads, self.output_dim).transpose(0, 1)
        ret = layer_norm(F.dropout(ret, p=self.dropout, training=self.training))
        # src = torch.cat((eq, wd), 0)
        # src_mask = torch.cat((eq_mask, wd_mask), 0)
        if return_src:
            return ret, eq, eq_mask
        return ret
    @classmethod
    def from_pretrained_projencoder(cls, num_proj_heads, vocab_eq, vocab_wd, model_args, ckpt):
        model = cls(num_proj_heads, vocab_eq, vocab_wd, model_args.layers, model_args.embed_dim, model_args.ff_embed_dim,
                    model_args.num_heads, model_args.dropout, model_args.output_dim)
        state_dict = torch.load(ckpt, map_location='cpu')
        model.eq_encoder.load_state_dict({k[len('eq_encoder.'):]: v for k, v in state_dict.items() if k.startswith('eq_encoder.')})
        # model.wd_encoder.load_state_dict({k[len('wd_encoder.'):]: v for k, v in state_dict.items() if k.startswith('wd_encoder.')})
        weight = state_dict['proj.weight'].repeat(num_proj_heads, 1)
        bias = state_dict['proj.bias'].repeat(num_proj_heads)
        model.proj.weight = nn.Parameter(weight)
        model.proj.bias = nn.Parameter(bias)
        return model

class MultiProjEncoder_wd(nn.Module):
    def __init__(self, num_proj_heads, vocab_eq, vocab_wd, layers, embed_dim, ff_embed_dim, num_heads, dropout, output_dim):
        super(MultiProjEncoder_wd, self).__init__()
        # self.eq_encoder = MonoEncoder(vocab_eq, layers, embed_dim, ff_embed_dim, num_heads, dropout)
        # self.eq_encoder = RnnEncoder(vocab_eq, embed_dim, 512, 2)
        self.wd_encoder = MonoEncoder(vocab_wd, layers, embed_dim, ff_embed_dim, num_heads, dropout)
        self.proj = nn.Linear(embed_dim, num_proj_heads*output_dim)
        self.num_proj_heads = num_proj_heads
        self.output_dim = output_dim
        self.dropout = dropout
    def forward(self, input_eq, input_wd, batch_first=False, return_src=False):
        if batch_first:
            input_wd = input_wd.t()
            # input_wd = input_wd.t()
        # eq, eq_mask = self.eq_encoder(input_eq)
        wd, wd_mask = self.wd_encoder(input_wd)
        ret = wd[0, :, :]
        # ret_wd = wd[0, :, :]
        # ret = (ret_eq + ret_wd) / 2
        ret = F.dropout(ret, p=self.dropout, training=self.training)
        ret = self.proj(ret).view(-1, self.num_proj_heads, self.output_dim).transpose(0, 1)
        ret = layer_norm(F.dropout(ret, p=self.dropout, training=self.training))
        # src = torch.cat((eq, wd), 0)
        # src_mask = torch.cat((eq_mask, wd_mask), 0)
        if return_src:
            return ret, wd, wd_mask
        return ret
    @classmethod
    def from_pretrained_projencoder(cls, num_proj_heads, vocab_eq, vocab_wd, model_args, ckpt):
        model = cls(num_proj_heads, vocab_eq, vocab_wd, model_args.layers, model_args.embed_dim, model_args.ff_embed_dim,
                    model_args.num_heads, model_args.dropout, model_args.output_dim)
        state_dict = torch.load(ckpt, map_location='cpu')
        model.wd_encoder.load_state_dict({k[len('wd_encoder.'):]: v for k, v in state_dict.items() if k.startswith('wd_encoder.')})
        # model.wd_encoder.load_state_dict({k[len('wd_encoder.'):]: v for k, v in state_dict.items() if k.startswith('wd_encoder.')})
        weight = state_dict['proj.weight'].repeat(num_proj_heads, 1)
        bias = state_dict['proj.bias'].repeat(num_proj_heads)
        model.proj.weight = nn.Parameter(weight)
        model.proj.bias = nn.Parameter(bias)
        return model

class MultiProjEncoder(nn.Module):
    def __init__(self, num_proj_heads, vocab, layers, embed_dim, ff_embed_dim, num_heads, dropout, output_dim):
        super(MultiProjEncoder, self).__init__()
        self.encoder = MonoEncoder(vocab, layers, embed_dim, ff_embed_dim, num_heads, dropout)
        self.proj = nn.Linear(embed_dim, num_proj_heads*output_dim)
        self.num_proj_heads = num_proj_heads
        self.output_dim = output_dim
        self.dropout = dropout
    def forward(self, input_ids, batch_first=False, return_src=False):
        if batch_first:
            input_ids = input_ids.t()
        src, src_mask = self.encoder(input_ids) 
        ret = src[0,:,:]
        ret = F.dropout(ret, p=self.dropout, training=self.training)
        ret = self.proj(ret).view(-1, self.num_proj_heads, self.output_dim).transpose(0, 1)
        ret = layer_norm(F.dropout(ret, p=self.dropout, training=self.training))
        if return_src:
            return ret, src, src_mask
        return ret
    @classmethod
    def from_pretrained_projencoder(cls, num_proj_heads, vocab, model_args, ckpt):
        model = cls(num_proj_heads, vocab, model_args.layers, model_args.embed_dim, model_args.ff_embed_dim, model_args.num_heads, model_args.dropout, model_args.output_dim)
        state_dict = torch.load(ckpt, map_location='cpu')
        model.encoder.load_state_dict({k[len('encoder.'):]:v for k,v in state_dict.items() if k.startswith('encoder.')})
        weight = state_dict['proj.weight'].repeat(num_proj_heads, 1)
        bias = state_dict['proj.bias'].repeat(num_proj_heads)
        model.proj.weight = nn.Parameter(weight)
        model.proj.bias = nn.Parameter(bias)
        return model

class ProjEncoder_spc(nn.Module):
    def __init__(self, vocab_eq, vocab_wd, layers, embed_dim, ff_embed_dim, num_heads, dropout, output_dim):
        super(ProjEncoder_spc, self).__init__()
        # self.eq_encoder = MonoEncoder(vocab_eq, layers, embed_dim, ff_embed_dim, num_heads, dropout)   # dolphin18k
        self.eq_encoder = RnnEncoder(vocab_eq, embed_dim, 512, 2)
        # self.wd_encoder = MonoEncoder(vocab_wd, layers, embed_dim, ff_embed_dim, num_heads, dropout)
        self.proj = nn.Linear(embed_dim, output_dim)
        self.dropout = dropout
        self.output_dim = output_dim
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.constant_(self.proj.bias, 0.)

    def forward(self, input_eq, input_wd, batch_first=False, return_src=False):
        if batch_first:
            input_eq = input_eq.t()
        eq, eq_mask = self.eq_encoder(input_eq)
        ret = torch.mean(eq, 0)
        ret = F.dropout(ret, p=self.dropout, training=self.training)
        ret = self.proj(ret)
        ret = layer_norm(ret)
        if return_src:
            return ret, eq, eq_mask
        return ret

    @classmethod
    def from_pretrained(cls, vocab_eq, vocab_wd, model_args, ckpt):
        model = cls(vocab_eq, vocab_wd, model_args.layers, model_args.embed_dim, model_args.ff_embed_dim, model_args.num_heads, model_args.dropout, model_args.output_dim)
        model.load_state_dict(torch.load(ckpt, map_location='cpu'))
        return model


class ProjEncoder(nn.Module):
    def __init__(self, vocab, layers, embed_dim, ff_embed_dim, num_heads, dropout, output_dim):
        super(ProjEncoder, self).__init__()
        self.encoder = MonoEncoder(vocab, layers, embed_dim, ff_embed_dim, num_heads, dropout)
        self.proj = nn.Linear(embed_dim, output_dim)
        self.dropout = dropout
        self.output_dim = output_dim
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.constant_(self.proj.bias, 0.)

    def forward(self, input_ids, batch_first=False, return_src=False):
        if batch_first:
            input_ids = input_ids.t()
        src, src_mask = self.encoder(input_ids)
        ret = src[0, :, :]
        # ret = torch.mean(src, 0)
        ret = F.dropout(ret, p=self.dropout, training=self.training)
        ret = self.proj(ret)
        ret = layer_norm(ret)
        if return_src:
            return ret, src, src_mask
        return ret

    @classmethod
    def from_pretrained(cls, vocab, model_args, ckpt):
        model = cls(vocab, model_args.layers, model_args.embed_dim, model_args.ff_embed_dim, model_args.num_heads, model_args.dropout, model_args.output_dim)
        model.load_state_dict(torch.load(ckpt, map_location='cpu'))
        return model

def batchify(data, vocab):

    tokens = [[BOS] + x for x in data]

    token = ListsToTensor(tokens, vocab)

    return token

class DataLoader(object):
    def __init__(self, used_data, vocab, batch_size, max_seq_len=256):
        self.vocab = vocab
        self.batch_size = batch_size

        data = []
        for x in used_data:
            x = x.split()[:max_seq_len]
            data.append(x)
        self.data = data

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        indices = np.arange(len(self))

        cur = 0
        while cur < len(indices):
            data = [self.data[i] for i in indices[cur:cur+self.batch_size]]
            cur += self.batch_size
            yield batchify(data, self.vocab)

@torch.no_grad()
def get_features(batch_size, norm_th, vocab, model, used_data, used_ids, max_norm=None, max_norm_cf=1.0):
    vecs, ids = [], []
    model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    model.eval()
    data_loader = DataLoader(used_data, vocab, batch_size)
    cur, tot = 0, len(used_data)
    for batch in asynchronous_load(data_loader):
        batch = move_to_device(batch, torch.device('cuda', 0)).t()
        bsz = batch.size(0)
        cur_vecs = model(batch, batch_first=True).detach().cpu().numpy()
        valid = np.linalg.norm(cur_vecs, axis=1) <= norm_th
        vecs.append(cur_vecs[valid])
        ids.append(used_ids[cur:cur+batch_size][valid])
        cur += bsz
        logger.info("%d / %d", cur, tot)
    vecs = np.concatenate(vecs, 0)
    ids = np.concatenate(ids, 0)
    out, max_norm = augment_data(vecs, max_norm, max_norm_cf)
    return out, ids, max_norm


class MatchingModel_wd(nn.Module):
    def __init__(self, query_encoder, response_encoder, bow=False):
        super(MatchingModel_wd, self).__init__()
        self.query_encoder = query_encoder
        self.response_encoder = response_encoder

    def forward(self, eq_orig, wd_orig, response, label_smoothing=0.):
        ''' query and response: [seq_len, batch_size]
        '''
        eq_len, bsz = eq_orig.size()
        q, q_src, _ = self.query_encoder(eq_orig, wd_orig, return_src=True)
        r, r_src, _ = self.response_encoder(response, return_src=True)
        q_src = q_src[0, :, :]
        r_src = r_src[0, :, :]
        scores = torch.mm(q, r.t())  # bsz x (bsz + adt)
        gold = torch.arange(bsz, device=scores.device)
        _, pred = torch.max(scores, -1)
        acc = torch.sum(torch.eq(gold, pred).float()) / bsz
        log_probs = F.log_softmax(scores, -1)
        loss, _ = label_smoothed_nll_loss(log_probs, gold, label_smoothing, sum=True)
        loss = loss / bsz
        if self.bow:

            loss_bow_wd = self.wd_bow(r_src, wd_orig.transpose(0, 1))
            loss_bow_r = self.response_bow(q_src, response.transpose(0, 1))

            loss = loss + loss_bow_wd + loss_bow_r

        return loss, acc, bsz
    def work(self, eq_orig, wd_orig, response):
        ''' query and response: [seq_len x batch_size ]
        '''
        _, bsz = eq_orig.size()
        q = self.query_encoder(eq_orig, wd_orig)
        r = self.response_encoder(response)
        scores = torch.sum(q * r, -1)
        return scores
    def save(self, model_args, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.query_encoder.state_dict(), os.path.join(output_dir, 'wd_query_encoder'))
        torch.save(self.response_encoder.state_dict(), os.path.join(output_dir, 'wd_response_encoder'))
        torch.save(model_args, os.path.join(output_dir, 'wd_args'))
    @classmethod
    def from_params(cls, vocabs, layers, embed_dim, ff_embed_dim, num_heads, dropout, output_dim, bow):
        query_encoder = ProjEncoder_wd(vocabs['eq_src'], vocabs['wd_src'], layers, embed_dim, ff_embed_dim, num_heads,
                                        dropout, output_dim)
        response_encoder = ProjEncoder(vocabs['tgt'], layers, embed_dim, ff_embed_dim, num_heads, dropout, output_dim)
        model = cls(query_encoder, response_encoder, bow)
        return model
    @classmethod
    def from_pretrained(cls, vocabs, input_dir):
        model_args = torch.load(os.path.join(input_dir, 'wd_args'))
        query_encoder = ProjEncoder_wd.from_pretrained(vocabs['eq_src'], vocabs['wd_src'], model_args,
                                                        os.path.join(input_dir, 'wd_query_encoder'))
        response_encoder = ProjEncoder.from_pretrained(vocabs['tgt'], model_args,
                                                       os.path.join(input_dir, 'wd_response_encoder'))
        model = cls(query_encoder, response_encoder)
        return model

class ProjEncoder_wd(nn.Module):
    def __init__(self, vocab_eq, vocab_wd, layers, embed_dim, ff_embed_dim, num_heads, dropout, output_dim):
        super(ProjEncoder_wd, self).__init__()

        self.wd_encoder = MonoEncoder(vocab_wd, layers, embed_dim, ff_embed_dim, num_heads, dropout)
        self.proj = nn.Linear(embed_dim, output_dim)
        self.dropout = dropout
        self.output_dim = output_dim
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.constant_(self.proj.bias, 0.)
    def forward(self, input_eq, input_wd, batch_first=False, return_src=False):
        if batch_first:

            input_wd = input_wd.t()

        wd, wd_mask = self.wd_encoder(input_wd)
        ret = wd[0, :, :]

        ret = F.dropout(ret, p=self.dropout, training=self.training)
        ret = self.proj(ret)
        ret = layer_norm(ret)

        if return_src:
            return ret, wd, wd_mask
        return ret
    @classmethod
    def from_pretrained(cls, vocab_eq, vocab_wd, model_args, ckpt):
        model = cls(vocab_eq, vocab_wd, model_args.layers, model_args.embed_dim, model_args.ff_embed_dim, model_args.num_heads, model_args.dropout, model_args.output_dim)
        model.load_state_dict(torch.load(ckpt, map_location='cpu'))
        return model