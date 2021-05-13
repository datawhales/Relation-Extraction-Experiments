import os
import json
from nltk.tokenize import WordPunctTokenizer
import argparse

def get_data(filename):
    data = []
    with open(filename) as f:
        all_lines = f.readlines()
        for line in all_lines:
            item = json.loads(line)
            data.append(item)
    return data

def write_data(filename, data):
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            itemdump = json.dumps(item)
            f.write(itemdump)
            f.write('\n')
# wiki80
def write_rel2id_and_type2id_for_wiki80(rel_filename, type_filename, data):
    rel2id = dict()
    type_list = []
    rel_idx = 0
    for item in data:
        if item['relation'] not in rel2id.keys():
            rel2id[item['relation']] = rel_idx
            rel_idx += 1
        if item['h']['type'] not in type_list:
            type_list.append(item['h']['type'])
        if item['t']['type'] not in type_list:
            type_list.append(item['t']['type'])

    type2id = dict()
    type2id['UNK'] = 0
    idx = 1
    for t in type_list:
        type2id['subj_' + t] = idx
        idx += 1
        type2id['obj_' + t] = idx
        idx += 1

    with open(rel_filename, 'w', encoding='utf-8') as f:
        json.dump(rel2id, f)
    with open(type_filename, 'w', encoding='utf-8') as f:
        json.dump(type2id, f)

def convert_kbp37(data):
    new_data = []
    for item in data:
        new_dict = dict()
        h_dict, t_dict = dict(), dict()
        
        tokens = item['token']
        new_dict['token'] = tokens
        
        h_dict['name'] = item['h']['name']
        t_dict['name'] = item['t']['name']
        
        h_tokens = item['h']['name'].split(' ')
        t_tokens = item['t']['name'].split(' ')
        
        h_pos, t_pos = [], []
        
        for h_token in h_tokens:
            try:
                h_pos.append(tokens.index(h_token))
            except:
                h_pos.append(0)
                
        for t_token in t_tokens:
            try:
                t_pos.append(tokens.index(t_token))
            except:
                t_pos.append(0)
                
        h_pos = [h_pos[0], h_pos[-1]+1]
        t_pos = [t_pos[0], t_pos[-1]+1]
        
        if h_pos[0] > h_pos[-1]:
            h_pos = [h_pos[-1], h_pos[0]]
        if t_pos[0] > t_pos[-1]:
            t_pos = [t_pos[-1], t_pos[0]]
        
        h_dict['pos'] = h_pos
        t_dict['pos'] = t_pos
        
        new_dict['h'] = h_dict
        new_dict['t'] = t_dict
        
        new_dict['relation'] = item['relation']
        
        new_data.append(new_dict)
    
    return new_data

# chemprot
def convert_chemprot(data):
    tokenizer = WordPunctTokenizer()
    new_data = []
    for item in data:
        new_dict = dict()
        h_dict, t_dict = dict(), dict()
        
        lst1 = item['text'].split("<<")
        lst2 = lst1[-1].split(">>")
        lst3 = lst2[-1].split("[[")
        lst4 = lst3[-1].split("]]")
        spe_added_text = " ".join([lst1[0], 'SPE0', lst2[0], 'SPE1', lst3[0], 'SPE2', lst4[0], 'SPE3', lst4[-1]])
        tokens = tokenizer.tokenize(spe_added_text)
        
        h_pos, t_pos = [], []
        spe_list = ['SPE0', 'SPE1', 'SPE2', 'SPE3']
        h_pos.append(tokens.index(spe_list[0]))
        h_pos.append(tokens.index(spe_list[1])-1)
        t_pos.append(tokens.index(spe_list[2])-2)
        t_pos.append(tokens.index(spe_list[3])-3)
        
        for spe in spe_list:
            tokens.remove(spe)
        
        new_dict['token'] = tokens
        h_dict['pos'] = h_pos 
        t_dict['pos'] = t_pos
        new_dict['h'] = h_dict
        new_dict['t'] = t_dict
        new_dict['relation'] = item['label']
    
        new_data.append(new_dict)
        
    return new_data

def write_rel2id(filename, data):
    rel2id = dict()
    rel_idx = 0
    for item in data:
        if item['relation'] not in rel2id.keys():
            rel2id[item['relation']] = rel_idx
            rel_idx += 1
        
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(rel2id, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dataset", dest="dataset", type=str,
                        default="chemprot", help="dataset to make rel2id.json and type2id.json")
    
    args = parser.parse_args()

    dev_data = get_data(os.path.join("../data", args.dataset, "dev.txt"))
    test_data = get_data(os.path.join("../data", args.dataset, "test.txt"))
    train_data = get_data(os.path.join("../data", args.dataset, "train.txt"))
    train_01_data = get_data(os.path.join("../data", args.dataset, "train_0.1.txt"))
    train_001_data = get_data(os.path.join("../data", args.dataset, "train_0.01.txt"))

    if args.dataset == "wiki80":
        write_rel2id_and_type2id_for_wiki80(os.path.join("../data", args.dataset, "rel2id.json"), 
                                            os.path.join("../data", args.dataset, "type2id.json"), train_data)

    else:
        if args.dataset == "chemprot":
            convert_function = convert_chemprot
        elif args.dataset == "kbp37":
            convert_function = convert_kbp37

        new_dev = convert_function(dev_data)
        new_test = convert_function(test_data)
        new_train = convert_function(train_data)
        new_train_01 = convert_function(train_01_data)
        new_train_001 = convert_function(train_001_data)
        
        write_data(os.path.join("../data", args.dataset, "new_dev.txt"), new_dev)
        write_data(os.path.join("../data", args.dataset, "new_test.txt"), new_test)
        write_data(os.path.join("../data", args.dataset, "new_train.txt"), new_train)
        write_data(os.path.join("../data", args.dataset, "new_train_0.1.txt"), new_train_01)
        write_data(os.path.join("../data", args.dataset, "new_train_0.01.txt"), new_train_001)

        write_rel2id(os.path.join("../data", args.dataset, "rel2id.json"), new_train)
