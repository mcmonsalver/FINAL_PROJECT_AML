from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from os.path import join
import re, os, json
import torch.nn as nn
import torch
from transformers import RobertaTokenizer, RobertaForMultipleChoice

from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule
from torch import optim
import argparse
from cytoolz import curry, concat
from torch.nn.utils import clip_grad_norm_
from tqdm import trange, tqdm
from tensorboardX import SummaryWriter
import time
from datetime import timedelta


class MultipleChoiceDataset(Dataset):
    
    """ get the article sentences only (for decoding use)"""
    def __init__(self, split: str, path: str) -> None:
        assert split in ['train', 'val', 'test']
        self.split = split
        self._data_path = join(path)
        self._n_data = _count_data(self._data_path, self.split)

    def __len__(self) -> int: #esto debe retornar la longitud del dataset (train = 39905, val = 10042) CHECK
        return self._n_data

    def __getitem__(self, i):
        
        with open(join(self._data_path, self.split + '.json')) as f:
            js_data = json.loads(f.read())
        context = js_data['ctx']
        answer, choice1, choice2, choice3, choice4 = js_data['label'], js_data['choice1'], js_data['choice2'], js_data['choice3'], js_data['choice4']

        return context, answer, [choice1, choice2, choice3, choice4]

def _count_data(path, split): #working!
    n_data = 0
    with open(join(path, split + '.json')) as f:
            js_data = json.loads(f.read())
    n_data += len(js_data['ctx'])

    f.close()
    return n_data


def batcher(path, bs):
    #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer = RobertaTokenizer.from_pretrained("roberta-large")    
    @curry
    def coll(tokenizer, batch):
        breakpoint()
        print('hola')
        contexts, answers, choicess = list(filter(bool, list(zip(*batch))))

        _inputs = [tokenizer([contexts[0][i], contexts[0][i], contexts[0][i], contexts[0][i]],
                  [choicess[0][0][i], choicess[0][1][i], choicess[0][2][i], choicess[0][3][i]], 
                  return_tensors='pt', padding=True)['input_ids'] for i in range(len(contexts[0]))]
        
        _inputs = pad_batch_tensorize_3d(_inputs, pad=0, cuda=False)

        return (_inputs), answers

    c = coll(tokenizer)

    #train_loader = DataLoader(MultipleChoiceDataset('train', path), batch_size=bs, shuffle=True, num_workers=4, collate_fn=coll(tokenizer), pin_memory=True)

    #test_loader = DataLoader(MultipleChoiceDataset('val', path), batch_size=bs, shuffle=False, num_workers=4, collate_fn=coll(tokenizer), pin_memory=True)

    return c #train_loader, test_loader

def pad_batch_tensorize_3d(inputs, pad, cuda=True):
    """pad_batch_tensorize
    :param inputs: List of size B containing torch tensors of shape [T, ...]
    :type inputs: List[np.ndarray]
    :rtype: TorchTensor of size (B, T, ...)
    """
    tensor_type = torch.cuda.LongTensor if cuda else torch.LongTensor
    batch_size = len(inputs)
    max_len = max([len(x) for _input in inputs for x in _input])
    if len(inputs) > 1:
        assert len(inputs[0]) == len(inputs[1])
    tensor_shape = (batch_size, len(inputs[0]), max_len)
    tensor = tensor_type(*tensor_shape)
    tensor.fill_(pad)
    for i, ids in enumerate(inputs):
        for j, _input in enumerate(ids):
            tensor[i, j, :len(_input)] = tensor_type(_input)
    return tensor


prueba = batcher('./', 3)
print('prueba', prueba)