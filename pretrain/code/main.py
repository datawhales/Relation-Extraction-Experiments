import os
import json
import torch
import torch.nn as nn

import sys
import argparse
import matplotlib
import pdb
import numpy as np 
import time
import random
import time
import matplotlib.pyplot as plt
matplotlib.use('Agg')
# from apex import amp
from torch.utils import data
from transformers import AdamW, get_linear_schedule_with_warmup
from dataset import *
from model import *

def log_loss(step_record, loss_record):
    if not os.path.exists("../img"):
        os.mkdir("../img")
    if not os.path.exists("../img/" + args.save_dir):
        os.mkdir("../img/" + args.save_dir)
    plt.plot(step_record, loss_record, lw=2)
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.title('loss curve')
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.savefig(os.path.join("../img", args.save_dir, 'loss_curve.png'))
    plt.close()

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def train(args, model, train_dataset):
    # total step
    step_tot = (len(train_dataset)  // args.gradient_accumulation_steps // args.batch_size_per_gpu // args.n_gpu) * args.max_epoch

    train_sampler = data.distributed.DistributedSampler(train_dataset) if args.local_rank != -1 else data.RandomSampler(train_dataset)
    params = {"batch_size": args.batch_size_per_gpu, "sampler": train_sampler}
    train_dataloader = data.DataLoader(train_dataset, **params)

    # # optimizer
    # no_decay = ['bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
    #     {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    #     ]
    # optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon, correct_bias=False)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=step_tot)

    if args.model == "MTB" or args.model == "CP" or args.model == "TRIPLE":
        # optimizer
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon, correct_bias=False)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=step_tot)
    elif args.model == "TS" or args.model == "TS_CP_SBERT":
        for name, param in model.named_parameters():
            if 'teacher_model' in name:
                param.requires_grad = False
        # optimizer
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon, correct_bias=False)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=step_tot)

    # amp training
    # model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    # distributed training
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    print("Begin train...")
    print("We will train model in %d steps" % step_tot)
    global_step = 0
    loss_record = []
    step_record = []
    for i in range(args.max_epoch):
        if args.local_rank != -1:    # Distributed training
            train_sampler.set_epoch(i)
        for step, batch in enumerate(train_dataloader):
        
            if args.model == "MTB":
                inputs = {"l_input": batch[0].to(args.device), "l_mask": batch[1].to(args.device),
                            "l_ph": batch[2].to(args.device), "l_pt": batch[3].to(args.device),
                            "r_input": batch[4].to(args.device), "r_mask": batch[5].to(args.device),
                            "r_ph": batch[6].to(args.device),"r_pt": batch[7].to(args.device),
                            "label": batch[8].to(args.device), "l_eh": batch[9].to(args.device),
                            "l_et": batch[10].to(args.device), "r_eh": batch[11].to(args.device),
                            "r_et": batch[12].to(args.device)}
            elif args.model == "CP" or args.model == "TS" or args.model == "TRIPLE":
                inputs = {"input": batch[0].to(args.device), "mask": batch[1].to(args.device),
                        "label": batch[2].to(args.device), "h_pos": batch[3].to(args.device),
                        't_pos': batch[4].to(args.device), "h_end": batch[5].to(args.device),
                        "t_end": batch[6].to(args.device)}
            elif args.model == "TS_CP_SBERT":
                inputs = {"input": batch[0].to(args.device), "mask": batch[1].to(args.device),
                        "label": batch[2].to(args.device), "h_pos": batch[3].to(args.device),
                        't_pos': batch[4].to(args.device), "h_end": batch[5].to(args.device),
                        "t_end": batch[6].to(args.device), "raw_text_id": batch[7].to(args.device)}

            model.train()
            m_loss, r_loss = model(**inputs)
            loss = m_loss + r_loss
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            
            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #     scaled_loss.backward()
            ########################
            # loss.sum().backward()
            if args.model == "MTB" or args.model == "CP" or args.model == "TRIPLE":
                # loss.sum().backward()
                loss.backward()
            elif args.model == "TS" or "TS_CP_SBERT":
                loss.backward()
            #########################
            
            if step % args.gradient_accumulation_steps == 0:
                # nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if args.local_rank in [0, -1] and global_step % args.save_step == 0:
                    if not os.path.exists("../ckpt"):
                        os.mkdir("../ckpt")
                    if not os.path.exists("../ckpt/"+args.save_dir):
                        os.mkdir("../ckpt/"+args.save_dir)
                    
                    # ckpt = {
                    #     'bert-base': model.module.model.bert.state_dict(),
                    # }
                    if args.model == "MTB" or args.model == "CP" or args.model == "TRIPLE":
                        ckpt = {
                            'bert-base': model.module.model.bert.state_dict(),
                        }
                    elif args.model == "TS":
                        ckpt = {
                            'bert-base': model.module.student_model.bert.state_dict(),
                        }
                    elif args.model == "TS_CP_SBERT":
                        model.module.student_model.save(os.path.join("../ckpt/" + args.save_dir, "ckpt_of_step_" + str(global_step)))
                    

                    # torch.save(ckpt, os.path.join("../ckpt/"+args.save_dir, "ckpt_of_step_"+str(global_step)))
                    if args.model != "TS_CP_SBERT":
                        torch.save(ckpt, os.path.join("../ckpt/"+args.save_dir, "ckpt_of_step_"+str(global_step)))

                if args.local_rank in [0, -1] and global_step % 5 == 0:
                    step_record.append(global_step)
                    loss_record.append(loss)
                
                if args.local_rank in [0, -1] and global_step % 500 == 0:
                    log_loss(step_record, loss_record)
                


                # if args.local_rank in [0, -1]:
                #     sys.stdout.write("step: %d, schedule: %.3f, mlm_loss: %.6f relation_loss: %.6f\r" % (global_step, global_step/step_tot, m_loss, r_loss))
                #     sys.stdout.flush()
                if args.model == "MTB" or args.model == "CP" or args.model == "TRIPLE":
                    if args.local_rank in [0, -1]:
                        sys.stdout.write("step: %d, schedule: %.3f, mlm_loss: %.6f relation_loss: %.6f\r" % (global_step, global_step/step_tot, m_loss, r_loss))
                        sys.stdout.flush()
                elif args.model == "TS" or args.model == "TS_CP_SBERT":
                    if args.local_rank in [0, -1]:
                        sys.stdout.write("step: %d, schedule: %.3f, teacher_student_loss: %.6f student_student_loss: %.6f\r" % (global_step, global_step/step_tot, m_loss, r_loss))
                        sys.stdout.flush()
                        
        
        # if args.train_sample:
        #     print("sampling...")
        #     train_dataloader.dataset.__sample__()
        #     print("sampled")
        if args.model != "TS_CP_SBERT":
            if args.train_sample:
                print("sampling...")
                train_dataloader.dataset.__sample__()
                print("sampled")

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
    
    parser.add_argument("--output_representation", dest="output_representation", type=str,
                        default="entity_marker", help="output representation {CLS, entity marker, all_markers, all_markers_concat, end_to_first, end_to_first_concat, marker_minus}")
    parser.add_argument("--anchor_method", dest="anchor_method", type=str,
                        default="user", help="anchor method {user, min, mean}")
    parser.add_argument("--user_length", dest="user_length", type=int,
                        default=7, help="user length used in anchor method")
    parser.add_argument("--anchor_feature", dest="anchor_feature", type=str,
                        default="one", help="use one anchor per relation or random or marker_dist")
    parser.add_argument("--sort_key", dest="sort_key", type=str,
                        default="random", help="sort key for anchors")
    parser.add_argument("--margin", dest="margin", type=float,
                        default=1.0, help="margin for triplet loss")         
    parser.add_argument("--portion", dest="portion", type=str,
                        default="20", help="portion of anchors")

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
        if not os.path.exists("../log/" + args.save_dir):
            os.mkdir("../log/" + args.save_dir)
        with open(os.path.join("../log/" + args.save_dir, "pretrain_log"), 'a+') as f:
            f.write(str(time.ctime())+"\n")
            f.write(str(args)+"\n")
            f.write("----------------------------------------------------------------------------\n")

    # Model and dataset
    if args.model == "MTB":
        model = MTB(args).to(args.device)
        train_dataset = MTBDataset("../data/MTB", args)
    elif args.model == "CP":
        model = CP(args).to(args.device)
        train_dataset = CPDataset("../data/CP", args)
    elif args.model == "TRIPLE":
        model = TRIPLE(args).to(args.device)
        train_dataset = TRIPLEDataset("../data/TRIPLE", args)
    elif args.model == "TS":
        model = TS(args).to(args.device)
        train_dataset = CPDataset("../data/CP", args)
    elif args.model == "TS_CP_SBERT":
        model = TS_CP_SBERT(args).to(args.device)
        train_dataset = CP_SBERT_Dataset("../data/CP", args)
    else:
        raise Exception("No such model! Please make sure that `model` takes the value in {MTB, CP, TS, TS_CP_SBERT}")

    # Barrier to make sure all process train the model simultaneously.
    if args.local_rank != -1:
        torch.distributed.barrier()
    train(args, model, train_dataset)