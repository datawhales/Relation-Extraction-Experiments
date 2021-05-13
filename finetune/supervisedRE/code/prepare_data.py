import os
import json
import argparse

def get_data(filename):
    data = []
    with open(filename) as f:
        all_lines = f.readlines()
        for line in all_lines:
            item = json.loads(line)
            data.append(item)
    return data

def write_rel2id_and_type2id(data):
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

    with open('rel2id.json', 'w', encoding='utf-8') as f:
        json.dump(rel2id, f)
    with open('type2id.json', 'w', encoding='utf-8') as f:
        json.dump(type2id, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dataset", dest="dataset", type=str,
                        default="chemprot", help="dataset to make rel2id.json and type2id.json")
    
    args = parser.parse_args()

    train_data = get_data(os.path.join("../data", args.dataset, "train.txt"))

    write_rel2id_and_type2id(train_data)
    