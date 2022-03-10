import os
import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
import random
import numpy as np
from jieba import analyse
import jieba.analyse

PAD, UNK, BOS, EOS = '<PAD>', '<UNK>', '<BOS>', '<EOS>'

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
    keywords = tfidt(text, keyword_num, allowPOS=('ns', 'n', 'nr', "vn"))
    return keywords

def is_number(uchar):
    """判断一个unicode是否是数字"""
    if uchar >= u'\u0030' and uchar <= u'\u0039':
        return True
    else:
        return False

def extract_wd(wd_list, wd_orig, wd_every):
    stop_words = ["his", "she", "he", "they", "you", "how", "How", "had", "did", "the", "of", "is", "and", "to", "in",
                  "that", "we", "for", "an", "are", "them",
                  "by", "be", "as", "on", "with", "can", "if", "from", "which", "you", "it",
                  "this", "then", "at", "have", "all", "not", "one", "has", "or", "that"]
    result = []
    for i, i1, i2 in zip(wd_list, wd_orig, wd_every):
        temp_every_wd = i1.copy()
        every_wd = i2.copy()
        for j in i:
            temp_str = ""
            for k in j:
                temp_str = temp_str + k
            keywords = tfidf_extract(temp_str, 2)     # for chinese

            # keywords1 = jieba.analyse.extract_tags(temp_str, 5)       # for english
            # result1 = keywords1.copy()
            # for key in keywords1:
            #     if is_number(key):
            #         result1.remove(key)
            #     if key in stop_words:
            #         result1.remove(key)
            # keywords = result1[:2]

            for h in keywords:
                if len(temp_every_wd) <= 6:
                    temp_every_wd.append(h)
                    for f in list(h):            # for chinese
                        every_wd.append(f)
                    # every_wd.append(h)  # for english
        every_wd = [BOS] + every_wd
        result.append(every_wd)
    return result






