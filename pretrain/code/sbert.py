from sentence_transformers import SentenceTransformer, models, evaluation, losses
from torch.utils.data import DataLoader, Dataset

import os
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
import sys
import argparse
import matplotlib
import time
import random
import matplotlib.pyplot as plt
matplotlib.use('Agg')

from tqdm import trange
from torch.utils import data
from collections import Counter
from transformers import AdamW, get_linear_schedule_with_warmup
from dataset import *



class CP_SBERT_Dataset(data.Dataset):
    """ Dataset for teacher-student model which uses CP as teacher model
        and SBERT as student model.  
    """
    def __init__(self, path, args):
        """ Inits tokenized sentence and pair of text with entity markers and raw text.

        Args:
            path: path to your dataset.
            args: args from command line.

        Returns:
            No returns.
        
        Raises:
            If the dataset in `path` is not the same format as described in
            file 'prepare_data.py', there may raise:
                - `key not found`
                - `integer can't be indexed`
                and so on.
        """
        self.path = path
        self.args = args
        # use same data with cp model
        data = json.load(open(os.path.join(path, "cpdata.json")))
        rel2scope = json.load(open(os.path.join(path, "rel2scope.json")))
        entityMarker = EntityMarker()

        self.tokens = np.zeros((len(data), args.max_length), dtype=int)
        self.mask = np.zeros((len(data), args.max_length), dtype=int)
        self.label = np.zeros((len(data)), dtype=int)
        self.h_pos = np.zeros((len(data)), dtype=int)
        self.t_pos = np.zeros((len(data)), dtype=int)
        self.h_end = np.zeros((len(data)), dtype=int)
        self.t_end = np.zeros((len(data)), dtype=int)

        self.raw_text_tokens = np.zeros((len(data), args.max_length), dtype=int)
        
        for i, rel in enumerate(rel2scope.keys()):
            scope = rel2scope[rel]
            for j in range(scope[0], scope[1]):
                self.label[j] = i

        for i, sentence in enumerate(data):
            h_flag = random.random() > args.alpha
            t_flag = random.random() > args.alpha
            h_p = sentence["h"]["pos"][0] 
            t_p = sentence["t"]["pos"][0]

            ids, ph, pt, eh, et = entityMarker.tokenize(sentence["tokens"], [h_p[0], h_p[-1]+1], [t_p[0], t_p[-1]+1], None, None, h_flag, t_flag)
            raw_text_ids = entityMarker.basic_tokenize(sentence["tokens"])

            length = min(len(ids), args.max_length)
            raw_length = min(len(raw_text_ids), args.max_length)
            self.tokens[i][:length] = ids[:length]
            self.mask[i][:length] = 1
            self.h_pos[i] = min(args.max_length-1, ph) 
            self.t_pos[i] = min(args.max_length-1, pt)
            self.h_end[i] = min(args.max_length-1, eh)
            self.t_end[i] = min(args.max_length-1, et)

            self.raw_text_tokens[i][:raw_length] = raw_text_ids[:raw_length]

    def __len__(self):
        """ Number of instances in an epoch.
        """
        return len(self.label)

    def __getitem__(self, index):
        """ Get training instance.

        Args:
            index: Instance index.

        Returns:
            input: Tokenized word id.
            mask: Attention mask for bert. 0 means masking, 1 means not masking.
            label: Label for sentence.
            h_pos: Position of head entity start.
            t_pos: Position of tail entity start.
            h_end: Position of head entity end.
            t_end: Position of tail entity end.
            raw_text_id: Raw text token ids.
        """
        input = self.tokens[index]
        mask = self.mask[index]
        label = self.label[index]
        h_pos = self.h_pos[index]
        t_pos = self.t_pos[index]
        h_end = self.h_end[index]
        t_end = self.t_end[index]

        raw_text_id = self.raw_text_tokens[index]

        return input, mask, label, h_pos, t_pos, h_end, t_end, raw_text_id


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="latentRE")
    parser.add_argument("--cuda", dest="cuda", type=str, 
                        default="4", help="gpu id")

    parser.add_argument("--lr", dest="lr", type=float,
                        default=5e-5, help='learning rate')
    parser.add_argument("--batch_size_per_gpu", dest="batch_size_per_gpu", type=int, 
                        default=32, help="batch size per gpu")
    parser.add_argument("--gradient_accumulation_steps", dest="gradient_accumulation_steps", type=int,
                        default=1, help="gradient accumulation steps")
    parser.add_argument("--max_epoch", dest="max_epoch", type=int, 
                        default=3, help="max epoch number")
    
    parser.add_argument("--alpha", dest="alpha", type=float,
                        default=0.3, help="true entity(not `BLANK`) proportion")

    parser.add_argument("--model", dest="model", type=str,
                        default="", help="{MTB, CP, TS}")
    parser.add_argument("--train_sample",action="store_true",
                        help="dynamic sample or not")
    parser.add_argument("--max_length", dest="max_length", type=int,
                        default=64, help="max sentence length")
    parser.add_argument("--bag_size", dest="bag_size", type=int,
                        default=2, help="bag size")
    parser.add_argument("--temperature", dest="temperature", type=float,
                        default=0.05, help="temperature for NTXent loss")
    parser.add_argument("--hidden_size", dest="hidden_size", type=int,
                        default=768, help="hidden size for mlp")
    
    parser.add_argument("--weight_decay", dest="weight_decay", type=float,
                        default=1e-5, help="weight decay")
    parser.add_argument("--adam_epsilon", dest="adam_epsilon", type=float,
                        default=1e-8, help="adam epsilon")
    parser.add_argument("--warmup_steps", dest="warmup_steps", type=int,
                        default=500, help="warmup steps")
    parser.add_argument("--max_grad_norm", dest="max_grad_norm", type=float,
                        default=1, help="max grad norm")
    
    parser.add_argument("--save_step", dest="save_step", type=int, 
                        default=10000, help="step to save")
    parser.add_argument("--save_dir", dest="save_dir", type=str,
                        default="", help="ckpt dir to save")
    #### revised
    parser.add_argument("--output_representation", dest="output_representation", type=str,
                        default="entity_marker", help="output representation {CLS, entity marker, all_markers, all_markers_concat, end_to_first, end_to_first_concat, marker_minus}")
    #### revised

    parser.add_argument("--teacher_model", dest="teacher_model", type=str,
                        default="../ckpt/ckpt_exp/end_to_first_concat/ckpt_of_step_60000", help="teacher model path")

    parser.add_argument("--pooling_method", dest="pooling_method", type=str,
                        default="mean", help="pooling method for entity marker representation after bert {mean, max, min}")

    parser.add_argument("--seed", dest="seed", type=int,
                        default=42, help="seed for network")

    parser.add_argument("--local_rank", dest="local_rank", type=int,
                        default=-1, help="local rank")
    args = parser.parse_args()

    # print args
    print('--------args----------')
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))
    print('--------args----------\n')
    
    # set backend
    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
    args.device = device
    args.n_gpu = len(args.cuda.split(","))
    set_seed(args)
    
    # log train
    if args.local_rank in [0, -1]:
        if not os.path.exists("../log"):
            os.mkdir("../log")
        with open("../log/pretrain_log", 'a+') as f:
            f.write(str(time.ctime())+"\n")
            f.write(str(args)+"\n")
            f.write("----------------------------------------------------------------------------\n")




    # Barrier to make sure all process train the model simultaneously.
    if args.local_rank != -1:
        torch.distributed.barrier()
    # train(args, model, train_dataset)
    teacher_model_ckpt = torch.load(args.teacher_model)
    teacher_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    model_dict = teacher_model.bert.state_dict()
    pretrained_dict = {k: v for k, v in teacher_model_ckpt['bert-base'].items() if k in model_dict}
    model_dict.update(pretrained_dict)
    teacher_model.bert.load_state_dict(model_dict)

    student_model_name = 'bert-base-uncased'
    word_embedding_model = models.Transformer(student_model_name, max_seq_length=args.max_length)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    student_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    ###################
    train_dataset = CP_SBERT_Dataset("../data/CP", args)
    
    ###################

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size_per_gpu)
    train_losses = losses.MSELoss(model=student_model)

    #----------------#
    # evaluator
    evaluators = []


    #----------------#