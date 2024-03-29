import torch
from torch import nn
import torch.nn.functional as F
import math
from stanfordcorenlp import StanfordCoreNLP
from decoding import CopyTokenDecoder
from transformer import Transformer, SinusoidalPositionalEmbedding, SelfAttentionMask, Embedding
from data import ListsToTensor, BOS, EOS, _back_to_txt_for_check
from search import Hypothesis, Beam, search_by_batch
from module import MonoEncoder, MonoEncoder_spc, MonoEncoder_spc_tr
from utils import extract_wd, move_to_device, extract_wd_en
import sys
sys.path.append('../MWPG-DMR')

class RetrieverGenerator(nn.Module):
    def __init__(self, vocabs, retriever, share_encoder,
                embed_dim, ff_embed_dim, num_heads, dropout, mem_dropout,
                enc_layers, dec_layers, mem_enc_layers, label_smoothing, datasets,
                segmentor, postagger, parser):
        super(RetrieverGenerator, self).__init__()
        self.vocabs = vocabs

        ####Retriever####
        self.share_encoder = share_encoder
        self.retriever = retriever
        self.encoder = MonoEncoder_spc(vocabs['eq_src'], vocabs['tgt'], enc_layers, embed_dim, ff_embed_dim, num_heads, dropout)
        # self.encoder = MonoEncoder(vocabs['tgt'], mem_enc_layers, embed_dim, ff_embed_dim, num_heads, mem_dropout)
        ####Retriever####

        self.tgt_embed = Embedding(vocabs['tgt'].size, embed_dim, vocabs['tgt'].padding_idx)
        self.tgt_pos_embed = SinusoidalPositionalEmbedding(embed_dim)
        self.decoder = Transformer(dec_layers, embed_dim, ff_embed_dim, num_heads, dropout, with_external=True)
        
        if share_encoder:
            self.mem_encoder = self.retriever.mem_feat_or_feat_maker.encoder
        else:
            self.mem_encoder = MonoEncoder(vocabs['tgt'], mem_enc_layers, embed_dim, ff_embed_dim, num_heads, mem_dropout)
        
        self.embed_scale = math.sqrt(embed_dim)
        self.self_attn_mask = SelfAttentionMask()
        self.output = CopyTokenDecoder(vocabs, self.tgt_embed, label_smoothing, embed_dim, ff_embed_dim, dropout)
        self.mem_bias_scale = nn.Parameter(torch.ones(retriever.num_heads))
        self.mem_bias_base = nn.Parameter(torch.zeros(retriever.num_heads))
        self.dropout = dropout
        self.datasets = datasets

        self.segmentor = segmentor
        self.postagger = postagger
        self.parser = parser
        # self.nlp = nlp

    ####Retriever####
    def retrieve_step(self, inp, work):
        #_back_to_txt_for_check(inp['tgt_tokens_in'], self.vocabs['tgt'])
        mem_ret = self.retriever.work(inp, allow_hit=work)
        return mem_ret
    ####Retriever####

    def encode_step(self, inp, work=False, update_mem_bias=True):
        # print("---------encode_step---------")
        mem_ret = self.retrieve_step(inp, work)

        if self.datasets == "math23k":
            wd_tokens = extract_wd(mem_ret['wd_retrieval_raw_sents'], inp['wd_orig'], inp['wd_every'], self.segmentor, self.postagger, self.parser)
        if self.datasets == "mawps":
            # print(self.datasets)
            # nlp = StanfordCoreNLP("/root/autodl-tmp/MWPG-DMR/stanford-corenlp-4.5.2")
            wd_tokens = extract_wd_en(mem_ret['wd_retrieval_raw_sents'], inp['wd_orig'], inp['wd_every'])
            # nlp.close()
        if self.datasets == "dolphin18k":
            wd_tokens = extract_wd_en(mem_ret['wd_retrieval_raw_sents'], inp['wd_orig'], inp['wd_every'])

        wd_token = torch.from_numpy(ListsToTensor(wd_tokens, self.vocabs['tgt']))
        wd_token = move_to_device(wd_token, inp['eq_tokens'].device)
        src_repr, src_mask = self.encoder(inp['eq_tokens'], wd_token)

        # src_repr, src_mask = self.encoder(inp['eq_tokens'], inp['wd_tokens'])

        inp.update(mem_ret)
        mem_repr, mem_mask = self.mem_encoder(inp['all_mem_tokens'])

        seq_len, _, dim = mem_repr.size()
        bsz = src_repr.size(1)
        mem_repr = mem_repr.view(-1, bsz, dim)
        mem_mask = mem_mask.view(-1, bsz)
        copy_seq = inp['all_mem_tokens'].view(-1, bsz)

        attn_bias = inp['all_mem_scores']
        if not update_mem_bias:
            attn_bias = attn_bias.detach()
        attn_bias = attn_bias * self.mem_bias_scale.view(1, -1, 1) + self.mem_bias_base.view(1, -1, 1)
        attn_bias = attn_bias.unsqueeze(0).expand(seq_len, -1, -1, -1).reshape(-1, bsz)

        return src_repr, src_mask, mem_repr, mem_mask, copy_seq, attn_bias

    def prepare_incremental_input(self, step_seq):
        token = torch.from_numpy(ListsToTensor(step_seq, self.vocabs['tgt']))
        return token

    def decode_step(self, step_token, state_dict, mem_dict, offset, topk): 
        src_repr = mem_dict['encoder_state']
        src_padding_mask = mem_dict['encoder_state_mask']
        mem_repr = mem_dict['mem_encoder_state']
        mem_padding_mask = mem_dict['mem_encoder_state_mask']
        copy_seq = mem_dict['copy_seq']
        mem_bias = mem_dict['mem_bias']

        _, bsz, _ = src_repr.size()

        new_state_dict = {}

        token_repr = self.embed_scale * self.tgt_embed(step_token) + self.tgt_pos_embed(step_token, offset)
        for idx, layer in enumerate(self.decoder.layers):
            name_i = 'decoder_state_at_layer_%d'%idx
            if name_i in state_dict:
                prev_token_repr = state_dict[name_i]
                new_token_repr = torch.cat([prev_token_repr, token_repr], 0)
            else:
                new_token_repr = token_repr

            new_state_dict[name_i] = new_token_repr
            token_repr, _, _ = layer(token_repr, kv=new_token_repr, external_memories=src_repr, external_padding_mask=src_padding_mask)
        name = 'decoder_state_at_last_layer'
        if name in state_dict:
            prev_token_state = state_dict[name]
            new_token_state = torch.cat([prev_token_state, token_repr], 0)
        else:
            new_token_state = token_repr
        new_state_dict[name] = new_token_state

        LL = self.output(token_repr, mem_repr, mem_padding_mask, mem_bias, copy_seq, None, work=True)

        def idx2token(idx, local_vocab):
            if (local_vocab is not None) and (idx in local_vocab):
                return local_vocab[idx]
            return self.vocabs['tgt'].idx2token(idx)

        topk_scores, topk_token = torch.topk(LL.squeeze(0), topk, 1) # bsz x k

        results = []
        for s, t in zip(topk_scores.tolist(), topk_token.tolist()):
            res = []
            for score, token in zip(s, t):
                res.append((idx2token(token, None), score))
            results.append(res)

        return new_state_dict, results

    @torch.no_grad()
    def work(self, data, beam_size, max_time_step, min_time_step=1):
        # print("--------work--------")
        src_repr, src_mask, mem_repr, mem_mask, copy_seq, mem_bias = self.encode_step(data, work=True)
        mem_dict = {'encoder_state':src_repr,
                    'encoder_state_mask':src_mask,
                    'mem_encoder_state':mem_repr,
                    'mem_encoder_state_mask':mem_mask,
                    'copy_seq':copy_seq,
                    'mem_bias':mem_bias}
        init_hyp = Hypothesis({}, [BOS], 0.)
        bsz = src_repr.size(1)
        beams = [ Beam(beam_size, min_time_step, max_time_step, [init_hyp]) for i in range(bsz)]
        search_by_batch(self, beams, mem_dict)
        return beams

    def forward(self, data, update_mem_bias=True):
        src_repr, src_mask, mem_repr, mem_mask, copy_seq, mem_bias = self.encode_step(data, update_mem_bias=update_mem_bias)
        tgt_in_repr = self.embed_scale * self.tgt_embed(data['tgt_tokens_in']) + self.tgt_pos_embed(data['tgt_tokens_in'])
        tgt_in_repr = F.dropout(tgt_in_repr, p=self.dropout, training=self.training)
        tgt_in_mask = torch.eq(data['tgt_tokens_in'], self.vocabs['tgt'].padding_idx)
        attn_mask = self.self_attn_mask(data['tgt_tokens_in'])

        tgt_out = self.decoder(tgt_in_repr,
                                  self_padding_mask=tgt_in_mask, self_attn_mask=attn_mask,
                                  external_memories=src_repr, external_padding_mask=src_mask)
        
        return self.output(tgt_out, mem_repr, mem_mask, mem_bias, copy_seq, data)
