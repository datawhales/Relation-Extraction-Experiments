import os
import re
import ast
import sys
sys.path.append('..')
import json
import pdb
import random
import numpy as np
import torch
sys.path.append('../../../')
from utils.utils import EntityMarker

class REDataset(torch.utils.data.Dataset):
    """ Data Loader for semeval, tacred.
    """
    def __init__(self, path, mode, args):
        super().__init__()
        
        data = []
        with open(os.path.join(path, mode)) as f:
            all_lines = f.readlines()
            for line in all_lines:
                item = json.loads(line)
                data.append(item)
        
        entityMarker = EntityMarker(args)
        total_instance = len(data)

        # load rel2id and type2id
        if os.path.exists(os.path.join(path, 'rel2id.json')):
            with open(os.path.join(path, 'rel2id.json')) as json_file:
                rel2id = json.load(json_file)
        else:
            raise Exception("Error: There is no 'rel2id.json' in " + path + ".")
        
        if os.path.exists(os.path.join(path, 'type2id.json')):
            with open(os.path.join(path, 'type2id.json')) as json_file:
                type2id = json.load(json_file)
        else:
            print("Warning: There is no 'type2id.json' in " + path + ".")
    
        print("preprocess " + mode)

        # preprocess data
        self.input_ids = np.zeros((total_instance, args.max_length), dtype=int)
        self.mask = np.zeros((total_instance, args.max_length), dtype=int)
        self.h_pos = np.zeros(total_instance, dtype=int)
        self.t_pos = np.zeros(total_instance, dtype=int)
        self.label = np.zeros(total_instance, dtype=int)
        #################### modified ######################
        self.h_end = np.zeros(total_instance, dtype=int)
        self.t_end = np.zeros(total_instance, dtype=int)
        #################### modified ######################

        for i, item in enumerate(data):
            self.label[i] = rel2id[item["relation"]]    # i+1 번째 문장의 relation id
            
            # tokenize
            if args.mode == "CM":
        #################### modified ######################
                ids, ph, pt, eh, et = entityMarker.tokenize(raw_text=item["token"], h_pos_range=item['h']['pos'], t_pos_range=item['t']['pos'])
        #################### modified ######################
            elif args.mode == "CT":
                h_type = f"[unused{type2id['subj_'+item['h']['type']] + 10}]"
                t_type = f"[unused{type2id['obj_'+item['t']['type']] + 10}]"
        #################### modified ######################
                ids, ph, pt, eh, et = entityMarker.tokenize(item["token"], item['h']['pos'], item['t']['pos'], h_type, t_type)
        #################### modified ######################
            else:
                raise Exception("No such mode! Please make sure that 'mode' takes the value in {CM, CT}")

            length = min(len(ids), args.max_length)
            self.input_ids[i][0:length] = ids[0:length]
            self.mask[i][0:length] = 1
            self.h_pos[i] = min(ph, args.max_length - 1)
            self.t_pos[i] = min(pt, args.max_length - 1)
        #################### modified ######################
            self.h_end[i] = min(eh, args.max_length - 1)
            self.t_end[i] = min(et, args.max_length - 1)
        #################### modified ######################
        print(f"The number of sentence in which tokenizer can't find head/tail entity is {entityMarker.err}")

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        input_ids = self.input_ids[index]
        mask = self.mask[index]
        h_pos = self.h_pos[index]
        t_pos = self.t_pos[index]
        label = self.label[index]
        h_end = self.h_end[index]
        t_end = self.t_end[index]

        return input_ids, mask, h_pos, t_pos, label, h_end, t_end, index