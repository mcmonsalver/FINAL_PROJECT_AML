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
import numpy as np
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

        context = js_data['ctx'][i]
        answer, choice1, choice2, choice3, choice4 = js_data['label'][i], js_data['choice1'][i], js_data['choice2'][i], js_data['choice3'][i], js_data['choice4'][i]

        return context, answer, [choice1, choice2, choice3, choice4]

def _count_data(path, split): #yes!
    n_data = 0

    with open(join(path, split + '.json')) as f:
            js_data = json.loads(f.read())

    n_data += len(js_data['ctx'])

    f.close()
    return n_data


def batcher(path, bs):
    tokenizer = RobertaTokenizer.from_pretrained("roberta-large")    
    @curry
    def coll(tokenizer, batch):

        #shape de batch es (bs, 3) ---> 3 siendo context, answer, [choices]
        #*batch separa los bs
        
        contexts, answers, choicess = list(filter(bool, list(zip(*batch))))
     
        _inputs = [tokenizer([contexts[i], contexts[i], contexts[i], contexts[i]],
                  [choicess[i][0], choicess[i][1], choicess[i][2], choicess[i][3]], 
                  return_tensors='pt', padding=True)['input_ids'] for i in range(len(contexts))]

        _inputs = pad_batch_tensorize_3d(_inputs, pad=0, cuda=False)

        return (_inputs), answers

    train_loader = DataLoader(MultipleChoiceDataset('train', path), batch_size=bs, collate_fn=coll(tokenizer), shuffle=True, num_workers=4)

    test_loader = DataLoader(MultipleChoiceDataset('val', path), batch_size=bs, collate_fn=coll(tokenizer),shuffle=False, num_workers=4)

    return train_loader, test_loader


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

class Bert_choice(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self._tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
        self._model =  RobertaForMultipleChoice.from_pretrained("roberta-large")
        self._device = device
        self._context = '[SEP]'
        self._choice = '[SEP]'
        self._choice_split = '[SEP]'

    def forward(self, _inputs, labels):
        outputs = self._model(_inputs, labels=labels)
        
        loss = outputs.loss
        classification_scores = outputs.logits
        _, _ids = torch.max(classification_scores, 1)
        return loss.mean(), _ids

    def evaluation(self, _inputs, labels):
        outputs = self._model(_inputs, labels=labels)
        
        loss = outputs.loss
        classification_scores = outputs.logits

        _, _ids = torch.max(classification_scores, 1)

        return classification_scores, _ids


def evaluate(model, loader, args):
    print('start validation: ')
    model.eval()
    total_ccr = 0
    total_loss = 0
    step = 0
    start = time.time()
    for _i, batch in enumerate(loader):
        with torch.no_grad():
            #questions, contexts, choicess = batch
            #_inputs = batch.to(args.device)

            _inputs = batch[0]
            _inputs = torch.tensor(_inputs).to(args.device)

            answers = batch[1]
            answers = torch.tensor(answers).to(args.device)

            bs, cn, length = _inputs.size()
            #labels = torch.tensor([0 for _ in range(bs)]).to(args.device)
            
            loss, _ids = model(_inputs, answers)
            #if args.n_gpu > 1:
            #    loss = loss.mean()  # mean() to average on multi-gpu parallel training
            
            ccr = sum([1 if _ids[i] == answers[i] else 0 for i in range(len(_ids))]) / len(_ids) 

            total_ccr += ccr
            total_loss += loss
            step += 1
    print('validation ccr: {:.4f} loss {:.4f}'.format(total_ccr / step, total_loss / step))
    print('validation finished in {} '.format(timedelta(seconds=int(time.time() - start))))
    model.train()
    return total_ccr / step, total_loss / step


def train(args):
    
    save_path = join(args.save_path, 'ckpt')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    train_loader, val_loader = batcher(args.path, args.bs)

    t_total = len(train_loader) // args.gradient_accumulation_steps * args.num_train_epochs
    print('1', t_total / args.num_train_epochs)
    tb_writer = SummaryWriter(log_dir=join(args.save_path, 'tensorboard'))

    model = Bert_choice()
    model = model.cuda()
    model.train()

    optimizer = AdamW(model.parameters(), lr=1e-5)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=0, t_total=t_total)
    
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    global_ccr = 0
    global_step = 0
    for _ in train_iterator:
        epoch_iterator = tqdm(train_loader, desc="Iteration")
        tr_loss, logging_loss = 0, 0
        tr_ccr = 0
        for step, batch in enumerate(epoch_iterator):
            #questions, contexts, choicess = batch
            #_inputs = batch.to(args.device)
             
            _inputs = batch[0]
            _inputs = torch.tensor(_inputs).to(args.device)

            answers = batch[1]
            answers = torch.tensor(answers).to(args.device)

            bs, cn, length = _inputs.size()
                        
            loss, _ids = model(_inputs, answers)
            #print(_ids, answers)
            ccr = sum([1 if _ids[i] == answers[i] else 0 for i in range(len(_ids))]) / len(_ids) 

               
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            #if args.n_gpu > 1:
            #    loss = loss.mean()  # mean() to average on multi-gpu parallel training

            tr_loss += loss.item()
            tr_ccr += ccr / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 2)
                optimizer.step()
                scheduler.step()
                model.zero_grad()

                global_ccr = global_ccr * 0.01 + tr_ccr
                tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                tb_writer.add_scalar('loss', (tr_loss - logging_loss), global_step)
                tb_writer.add_scalar('ccr', global_ccr, global_step)
                global_step += 1
                #print('loss: {:.4f} ccr {:.4f}\r'.format(tr_loss, ccr), end='')
                logging_loss = tr_loss
                tr_ccr = 0
                if global_step % args.ckpt == 0:
                    total_ccr, total_loss = evaluate(model, val_loader, args)
                    name = 'ckpt-{:4f}-{:4f}-{}'.format(total_loss, total_ccr, global_step)
                    save_dict = {}
                    save_dict['state_dict'] = model.state_dict()
                    save_dict['optimizer'] = optimizer.state_dict()
                    torch.save(save_dict, join(save_path, name))






if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=('combine an extractor and an abstractor '
                     'to decode summary and evaluate on the '
                     'CNN/Daily Mail dataset'))
    parser.add_argument('--path', required=True, help='path of data')
    parser.add_argument('--save_path', required=True, help='path to store/eval')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--no-cuda', action='store_true',
                        help='disable GPU training')
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--bs', type=int, action='store', default=5,
                        help='the training batch size')
    parser.add_argument('--num_train_epochs', type=int, action='store', default=10,
                        help='the training batch size')
    parser.add_argument('--ckpt', type=int, action='store', default=10000,
                        help='ckpt per global step')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    #torch.cuda.set_device(args.gpu_id)
    args.n_gpu = torch.cuda.device_count()
    print('use {} gpus'.format(args.n_gpu))
    args.device = 'cuda'
    train(args)