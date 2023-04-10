import os
import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
import random
import re
from stanfordcorenlp import StanfordCoreNLP
import numpy as np
from fnmatch import fnmatch
# from jieba import analyse
# import jieba.analyse
from pyltp import SentenceSplitter, Segmentor, Postagger, Parser, NamedEntityRecognizer, SementicRoleLabeller

PAD, UNK, BOS, EOS = '<PAD>', '<UNK>', '<BOS>', '<EOS>'
sd_token = ['SBV', 'VOB', 'FOB', 'IOB', 'ATT']
ld_token = ['HED', 'ADV', 'CMP', 'RAD']
rubbish_token = ['(', ')', '%', ',', '，', '.', '。', '？', '?']
wd_rubbish_token = ["元", "多少", "每", "它", "了", "他", "她", "它",
                    "几", "元", "千米", "这", "个",
                    "我", "千", "只", "当"]

en_sd_token = ['nsubj', 'obj', 'iobj', 'obl', 'NN', 'NNS', 'NNP', 'NNPS']
en_sd_rubbish_token = ['how', 'many', 'much', 'what', 'How', 'What', 'times']
en_ld_token = ['conj', 'advmod', 'mark', 'VB', 'VBD', 'VBG', 'VBN', 'VBP',
            'VBZ', 'IN', 'DT', 'CC', 'JJR']
en_ld_useful_token = ['how', 'many', 'much', 'what', 'How', 'What', 'times']
def partially_load(model, ckpt):
    pretrained_dict = torch.load(ckpt)
    model_dict = model.state_dict()
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

def move_to_device(maybe_tensor, device):
    if torch.is_tensor(maybe_tensor):
        return maybe_tensor.to(device)
    elif isinstance(maybe_tensor, np.ndarray):
        return torch.from_numpy(maybe_tensor).to(device).contiguous()
    elif isinstance(maybe_tensor, dict):
        return {
            key: move_to_device(value, device)
            for key, value in maybe_tensor.items()
        }
    elif isinstance(maybe_tensor, list):
        return [move_to_device(x, device) for x in maybe_tensor]
    elif isinstance(maybe_tensor, tuple):
        return tuple([move_to_device(x, device) for x in maybe_tensor])
    return maybe_tensor

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= size

class Statistics:
    def __init__(self, key_value_dict=None, **kwargs):
        self.statistics = {'steps':0}
        if key_value_dict is not None:
            for x in key_value_dict:
                self.statistics[x] = key_value_dict[x]
        for x in kwargs:
            self.statistics[x] = kwargs[x]

    def update(self, key_or_dict, value=None):
        if value is None:
            assert isinstance(key_or_dict, dict)
            for key in key_or_dict:
                if key not in self.statistics:
                    self.statistics[key] = 0.
                self.statistics[key] += key_or_dict[key]
        else:
            assert isinstance(key_or_dict, str)
            if key_or_dict not in self.statistics:
                self.statistics[key_or_dict] = 0.
            self.statistics[key_or_dict] += value
    
    def __getitem__(self, attr):
        return self.statistics[attr]

    def step(self):
        self.statistics['steps'] += 1

def data_proc(data, queue):
    for x in data:
        queue.put(x)
    queue.put('EPOCHDONE')

def asynchronous_load(data_loader):
    queue = mp.Queue(10)
    data_generator = mp.Process(target=data_proc, args=(data_loader, queue))
    data_generator.start()
    done = False
    while not done:
        batch = queue.get()
        if isinstance(batch, str):
            done = True
        else:
            yield batch
    data_generator.join()

def tfidf_extract(text, keyword_num):
    tfidt = analyse.extract_tags
    # analyse.set_stop_words('stopword.txt')
    keywords = tfidt(text, keyword_num, allowPOS=('a', 'n', 'nr', "ns", "nt", "nz", 'vn'))
    return keywords

def is_alphabet(uchar):
    """判断一个unicode是否是英文字母"""
    if (uchar >= u'\u0041' and uchar <= u'\u005a') or (uchar >= u'\u0061' and uchar <= u'\u007a'):
        return True
    else:
        return False

def yier(w):
    if fnmatch(w, "*一*") or fnmatch(w, "*二*") or fnmatch(w, "米") or fnmatch(w, "*克*"):
        return False
    else:
        return True

def is_number(uchar_orig):
    """判断一个unicode是否是数字"""
    uchar = uchar_orig[0]
    if uchar >= u'\u0030' and uchar <= u'\u0039':
        return True
    else:
        return False

def extract_wd(wd_list, wd_orig, wd_every, segmentor, postagger, parser):
    # print("----------extract_wd-----------")
    result = []
    for i, i1, i2 in zip(wd_list, wd_orig, wd_every):
        temp_every_wd = i1.copy()
        every_wd = i2.copy()
        temp_str1 = ""
        for j in i[0]:
            temp_str1 = temp_str1 + j
        temp_str2 = ""
        for j in i[1]:
            temp_str2 = temp_str2 + j
        temp_str3 = ""
        for j in i[2]:
            temp_str3 = temp_str3 + j
        keywords1, _ = lpt_extract(temp_str1, segmentor, postagger, parser)
        keywords2, _ = lpt_extract(temp_str2, segmentor, postagger, parser)
        keywords3, _ = lpt_extract(temp_str3, segmentor, postagger, parser)
        comlist_temp = [val for val in keywords1 if val in keywords2]
        comlist = [val for val in comlist_temp if val in keywords3]
        diflist = [val for val in comlist if val not in temp_every_wd]

        for h in diflist:
            temp_every_wd.append(h)
            for f in list(h):
                every_wd.append(f)

        every_wd = [BOS] + every_wd
        result.append(every_wd)
    return result

def lpt_extract(text, segmentor, postagger, parser):
    sentence_orig = SentenceSplitter.split(text)[0]
    #---------------------  分句 ------------------------
    sentence_list = sentence_orig.strip().split("，")
    sd_list = []
    ld_list = []
    # --------------------- 分词 ------------------------
    # segmentor = Segmentor(os.path.join(MODELDIR, "cws.model"))
    for sentence in sentence_list:
        words = segmentor.segment(sentence)
    # --------------------- 词性标注 ------------------------
    # postagger = Postagger(os.path.join(MODELDIR, "pos.model"))
        postags = postagger.postag(words)
    # --------------------- 语义依存分析 ------------------------
    # parser = Parser(os.path.join(MODELDIR, "parser.model"))
        arcs = parser.parse(words, postags)
        res = []
        for (head, relation) in arcs:
            res.append(relation)
    # print("\t".join("%s:%s" % (head, relation) for (head, relation) in zip(words, res)))
    #     index = 0
        for w, r in zip(words, res):
            if is_alphabet(w) or is_number(w) or w in rubbish_token:
                # index = index + 1
                continue
            else:
                if r in sd_token and w not in wd_rubbish_token and yier(w):
                    sd_list.append(w)
                    continue
                    # index = index + 1
                    # continue
                if r in ld_token:
                    ld_list.append(w)
                    # index = index + 1
                    continue
    return list(set(sd_list)), list(set(ld_list))

def extract_wd_en(wd_list, wd_orig, wd_every):
    result = []
    for i, i1, i2 in zip(wd_list, wd_orig, wd_every):
        temp_every_wd = i1.copy()
        every_wd = i2.copy()
        temp_str1 = ""
        for j in i[0]:
            temp_str1 = temp_str1 + j + " "
        temp_str2 = ""
        for j in i[1]:
            temp_str2 = temp_str2 + j + " "
        temp_str3 = ""
        for j in i[2]:
            temp_str3 = temp_str3 + j + " "

        keywords1, _, _ = stanford(temp_str1)
        keywords2, _, _ = stanford(temp_str2)
        keywords3, _, _ = stanford(temp_str3)

        comlist_temp = [val for val in keywords1 if val in keywords2]
        comlist = [val for val in comlist_temp if val in keywords3]
        diflist = [val for val in comlist if val not in temp_every_wd]

        for h in diflist:
            every_wd.append(h)

        every_wd = [BOS] + every_wd
        result.append(every_wd)
    return result

def takethird(ele):
    return ele[2]

def is_number_2(uchar_orig):
    uchar = uchar_orig[0]
    if uchar >= u'\u0030' and uchar <= u'\u0039':
        return True
    else:
        return False

def stanford(sentence_orig):
    # nlp = StanfordCoreNLP("/root/autodl-tmp/MWPG-DMR/stanford-corenlp-4.5.2")
    sentence_list = re.split("[,.]", sentence_orig)
    sd_list = []
    ld_list = []
    sentence_list_all = []
    for sentence in sentence_list:
        ld_list_every = []
        words = nlp.word_tokenize(sentence)
        tags_orig = nlp.pos_tag(sentence)
        tags = []
        for i in tags_orig:
            tags.append(i[1])
        deps_orig = nlp.dependency_parse(sentence)
        # nlp.close()
        deps_orig.sort(key=takethird)
        deps = []
        for i in deps_orig:
            deps.append(i[0])
        for w, t, d in zip(words, tags, deps):
            if is_number_2(w):
                continue
            else:
                if (t in en_sd_token or d in en_sd_token) and w not in en_sd_rubbish_token:
                    sd_list.append(w)
                    continue
                if t in en_ld_token or d in en_ld_token or w in en_ld_useful_token:
                    ld_list_every.append(w)
                    ld_list.append(w)
                    continue
        sentence_list_every = []
        for i in words:
            if is_number_2(i):
                sentence_list_every.append(i)
                continue
            if i in ld_list_every:
                sentence_list_every.append(i)
                continue
            else:
                sentence_list_every.append(" @ ")
        sen_str = ""
        for i in sentence_list_every:
            sen_str = sen_str + i + " "
        sentence_list_all.append(sen_str)
    sentence_res = ""
    for i in sentence_list_all[0:len(sentence_list_all) - 1]:
        sentence_res = sentence_res + i + "," + " "
    sentence_res = sentence_res + sentence_list_all[-1] + " " + "?"
    strAfter = re.sub(' +', ' ', sentence_res)
    # nlp.close()
    return list(set(sd_list)), strAfter, list(set(ld_list))
