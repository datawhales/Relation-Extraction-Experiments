import json 
import random
import os 
import sys 
sys.path.append("..")
import numpy as np  
from torch.utils.data import Dataset
sys.path.append("../../")
from utils.utils import EntityMarker


class TRIPLEDataset(Dataset):
    """ Dataset for triple model.
    This class prepare data for training of triple.
    """
    def __init__(self, path, args):
        """Inits tokenized sentence and [anchor, positive, negative] triplets for triple model.
        
        Args:
            path: path to your dataset.
            args: args from command line.
        
        Returns:
            No returns
        
        Raises:
            If the dataset in `path` is not the same format as described in 
            file 'prepare_data.py', there may raise:
                - `key not found`
                - `integer can't be indexed`
                and so on.
        """
        self.path = path
        self.args = args
        data = json.load(open(os.path.join(path, "tripledata.json")))
        rel2scope = json.load(open(os.path.join(path, "rel2scope.json")))
        entityMarker = EntityMarker()

        self.tokens = np.zeros((len(data), args.max_length), dtype=int)
        self.mask = np.zeros((len(data), args.max_length), dtype=int)
        self.label = np.zeros((len(data)), dtype=int)
        self.h_pos = np.zeros((len(data)), dtype=int)
        self.t_pos = np.zeros((len(data)), dtype=int)
        self.h_end = np.zeros((len(data)), dtype=int)
        self.t_end = np.zeros((len(data)), dtype=int)

        # Distant supervised label for sentence.
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

            length = min(len(ids), args.max_length)
            self.tokens[i][:length] = ids[:length]
            self.mask[i][:length] = 1
            self.h_pos[i] = min(args.max_length-1, ph)
            self.t_pos[i] = min(args.max_length-1, pt)
            self.h_end[i] = min(args.max_length-1, eh)
            self.t_end[i] = min(args.max_length-1, et)
            
        print(f"The number of sentence in which tokenizer can't find head/tail entity is {entityMarker.err}")

        # samples triplets
        self.__sample__()

    def __get_mean_token_length_of_each_rel__(self):
        """ To generate anchor sentences, find out mean length of sentences in each relation.
        """
        data = json.load(open(os.path.join(self.path, "tripledata.json")))
        rel2scope = json.load(open(os.path.join(self.path, "rel2scope.json")))
        mean_dict = dict()
        for key in rel2scope.keys():
            scope = rel2scope[key]
            mean = 0
            rel_num = scope[1] - scope[0]
            for i in range(scope[0], scope[1]):
                mean += len(data[i]['tokens'])
            mean = mean / rel_num
            mean_dict[key] = mean
        return mean_dict

    def __get_min_token_length_of_each_rel__(self):
        """ To generate anchor sentences, find out min length of sentences in each relation.
        """
        data = json.load(open(os.path.join(self.path, "tripledata.json")))
        rel2scope = json.load(open(os.path.join(self.path, "rel2scope.json")))
        min_dict = dict()
        for key in rel2scope.keys():
            scope = rel2scope[key]
            for i in range(scope[0], scope[1]):
                if i == scope[0]:
                    min_len = len(data[i]['tokens'])
                    continue
                if len(data[i]['tokens']) > min_len:
                    min_len = len(data[i]['tokens'])
            min_dict[key] = min_len
        return min_dict            

    def __get_user_token_length_of_each_rel__(self, length):
        """ To generate anchor sentences, make dictionary of specific length user choosed.
        """
        data = json.load(open(os.path.join(self.path, "tripledata.json")))
        rel2scope = json.load(open(os.path.join(self.path, "rel2scope.json")))
        user_dict = dict()
        for key in rel2scope.keys():
            scope = rel2scope[key]
            user_dict[key] = length
        return user_dict

    def __get_anchors__(self):
        """ Generate anchor sentences.
        Use the sentence with the minimum difference as the anchor sentence.
        """
        data = json.load(open(os.path.join(self.path, "tripledata.json")))
        rel2scope = json.load(open(os.path.join(self.path, "rel2scope.json")))
        anchor_dict = dict()

        if self.args.anchor_method == "mean":
            standard_dict = self.__get_mean_token_length_of_each_rel__()
        elif self.args.anchor_method == "min":
            standard_dict = self.__get_min_token_length_of_each_rel__()
        elif self.args.anchor_method == "user":
            user_length = self.args.user_length
            standard_dict = self.__get_user_token_length_of_each_rel__(user_length)
        
        for key in rel2scope.keys():
            scope = rel2scope[key]
            for i in range(scope[0], scope[1]):
                diff = abs(standard_dict[key] - len(data[i]['tokens']))
                if i == scope[0]:
                    min_diff = diff
                    anchor_dict[key] = [i, data[i]['tokens']]
                else:
                    if diff < min_diff:
                        min_diff = diff
                        anchor_dict[key] = [i, data[i]['tokens']]
        return anchor_dict
    
    def __sample__(self):
        """Samples triplets.
        After sampling, `self.triplet` is all pairs sampled.
        `self.triplet` example: 
                [
                    [0, 1, 674723],
                    [0, 2, 739671],
                    [0, 3, 47562],
                    [5, 4, 197608],
                    ...
                ]
        """
        data = json.load(open(os.path.join(self.path, "tripledata.json")))
        rel2scope = json.load(open(os.path.join(self.path, "rel2scope.json")))
        rel_key_list = list(rel2scope.keys())

        if self.args.anchor_feature == "random":
            self.triplet = []
            for i, key in enumerate(rel2scope.keys()):
                scope = rel2scope[key]
                
                pos_scope = list(range(scope[0], scope[1]))
                random.shuffle(pos_scope)
                
                bag = []
                for i, index in enumerate(pos_scope):
                    bag.append(index)
                    if i % 2 == 1:
                        neg_rel_index = random.sample(range(len(rel2scope)), 1)[0]
                        
                        if neg_rel_index == i:
                            while neg_rel_index == i:
                                neg_rel_index = random.sample(range(len(rel2scope)), 1)[0]
                        neg_key = rel_key_list[neg_rel_index]
                        neg_scopes = rel2scope[neg_key]
                        neg_rel = random.sample(range(neg_scopes[0], neg_scopes[1]), 1)[0]
                        
                        bag.append(neg_rel)
                        
                        self.triplet.append(bag)
                        bag = []

        elif self.args.anchor_feature == "marker_dist":
            self.triplet = []
            total_list = list(range(len(data)))
            
            for i, key in enumerate(rel2scope.keys()):
                pair_lst = []
                scope = rel2scope[key]
                pos_scope = list(range(scope[0], scope[1]))
                # sort by entity marker distance
                pos_scope.sort(key=lambda x: abs(data[x]['t']['pos'][0][0] - data[x]['h']['pos'][0][0]))
                # set anchor 10% of each relation
                gold_anchor_list = pos_scope[:len(pos_scope) // 10]
                gold_positive_list = pos_scope[len(pos_scope) // 10:]
                gold_positive_list = [x for x in gold_positive_list if x in total_list]
                # 90% shuffle
                random.shuffle(gold_positive_list)
                
                if not gold_anchor_list:
                    anchor_list = gold_positive_list[:len(gold_positive_list) // 2]
                    positive_list = gold_positive_list[len(gold_positive_list) // 2:]
                    if len(positive_list) > len(anchor_list):
                        positive_list = positive_list[:-1]
                        
                    anchor_num = len(anchor_list)
                        
                    left = [x for x in total_list if x not in range(scope[0], scope[1])]

                    if len(left) < anchor_num:
                        total_list = list(range(len(data)))
                        
                    left = [x for x in total_list if x not in range(scope[0], scope[1])]
                    
                    negative_list = random.sample(left, anchor_num)
                    
                    for index in range(anchor_num):
                        anchor = anchor_list[index]
                        positive = positive_list[index]
                        negative = negative_list[index]
                        
                        if anchor in total_list:
                            total_list.remove(anchor)
                        if positive in total_list:
                            total_list.remove(positive)
                        if negative in total_list:
                            total_list.remove(negative)
                    
                        pair_lst.append([anchor, positive, negative])
                        
                    self.triplet.extend(pair_lst)
                    
                # gold anchor exists
                else:
                    random.shuffle(gold_anchor_list)
                    
                    anchor_num = len(gold_anchor_list) // 2
                        
                    left = [x for x in total_list if x not in range(scope[0], scope[1])]
                    
                    if len(left) < anchor_num:
                        total_list = list(range(len(data)))
                        
                    left = [x for x in total_list if x not in range(scope[0], scope[1])]
                    
                    negative_list = random.sample(left, anchor_num)
                    
                    bag = []
                    for i, anchor in enumerate(gold_anchor_list):
                        bag.append(anchor)
                        if i % 2 == 1:
                            negative = negative_list[i // 2]
                            bag.append(negative)              

                            if negative in total_list:
                                total_list.remove(negative)
                                
                            pair_lst.append(bag)
                            bag = []
                    
                    self.triplet.extend(pair_lst)
                    
                    # not gold_anchors
                    anchor_list = gold_anchor_list * (len(gold_positive_list) // len(gold_anchor_list) + 1)
                    random.shuffle(anchor_list)
                    positive_list = gold_positive_list[:]

                    anchor_num = min(len(anchor_list), len(positive_list))
                    
                    left = [x for x in total_list if x not in range(scope[0], scope[1])]

                    if len(left) < anchor_num:
                        total_list = list(range(len(data)))
                    
                    left = [x for x in total_list if x not in range(scope[0], scope[1])]
                    
                    negative_list = random.sample(left, anchor_num)
                    
                    for index in range(anchor_num):
                        anchor = anchor_list[index]
                        positive = positive_list[index]
                        negative = negative_list[index]
                        
                        if anchor in total_list:
                            total_list.remove(anchor)
                        if positive in total_list:
                            total_list.remove(positive)
                        if negative in total_list:
                            total_list.remove(negative)
                    
                        pair_lst.append([anchor, positive, negative])
                    
                    self.triplet.extend(pair_lst)
    
        else:
            anchor_dict = self.__get_anchors__()

            self.triplet = []
            for index, key in enumerate(rel_key_list):
                scope = rel2scope[key]
                for i in range(scope[0], scope[1]):
                    if i == anchor_dict[key][0]:
                        continue

                    neg_rel_index = random.sample(range(len(rel2scope)), 1)[0]

                    if neg_rel_index == index:
                        while neg_rel_index == index:
                            neg_rel_index = random.sample(range(len(rel2scope)), 1)[0]
                    
                    neg_key = rel_key_list[neg_rel_index]

                    neg_scopes = rel2scope[neg_key]
                    neg_rel = random.sample(range(neg_scopes[0], neg_scopes[1]), 1)[0]

                    # # to sample negative pair which has similar length of tokens with positive pair
                    # cnt = 0
                    # if abs(len(data[i]['tokens']) - len(data[neg_rel]['tokens'])) > 5:
                    #     while abs(len(data[i]['tokens']) - len(data[neg_rel]['tokens'])) > 5:
                    #         neg_rel = random.sample(range(neg_scopes[0], neg_scopes[1]), 1)[0]
                    #         cnt += 1
                    #     if cnt >= (scope[1] - scope[0]):
                    #         break
                    # if abs(len(data[i]['tokens']) - len(data[neg_rel]['tokens'])) <= 5:
                    #     self.triplet.append([anchor_dict[key][0], i, neg_rel])
                    self.triplet.append([anchor_dict[key][0], i, neg_rel])

        print(f"The number of triplets is {len(self.triplet)}")

    def __len__(self):
        """ Number of instances in an epoch.
        """
        return len(self.triplet)

    def __getitem__(self, index):
        """ Get training instance.

        Args:
            index: Instance index.

        Return:
            
        """
        bag = self.triplet[index]
        input = np.zeros((self.args.max_length * 3), dtype=int)
        mask = np.zeros((self.args.max_length * 3), dtype=int)
        label = np.zeros((3), dtype=int)
        h_pos = np.zeros((3), dtype=int)
        t_pos = np.zeros((3), dtype=int)
        h_end = np.zeros((3), dtype=int)
        t_end = np.zeros((3), dtype=int)
        
        for i, ind in enumerate(bag):
            input[i * self.args.max_length : (i+1) * self.args.max_length] = self.tokens[ind]
            mask[i * self.args.max_length : (i+1) * self.args.max_length] = self.mask[ind]
            label[i] = self.label[ind]
            h_pos[i] = self.h_pos[ind]
            t_pos[i] = self.t_pos[ind]
            h_end[i] = self.h_end[ind]
            t_end[i] = self.t_end[ind]
        
        return input, mask, label, h_pos, t_pos, h_end, t_end

class CP_SBERT_Dataset(Dataset):
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

class CPDataset(Dataset):
    """Overwritten class Dataset for model CP.
    This class prepare data for training of CP.
    """
    def __init__(self, path, args):
        """Inits tokenized sentence and positive pair for CP.
        
        Args:
            path: path to your dataset.
            args: args from command line.
        
        Returns:
            No returns
        
        Raises:
            If the dataset in `path` is not the same format as described in 
            file 'prepare_data.py', there may raise:
                - `key nor found`
                - `integer can't be indexed`
                and so on.
        """
        self.path = path 
        self.args = args 
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

        # Distant supervised label for sentence.
        # Sentences whose label are the same in a batch 
        # is positive pair, otherwise negative pair.
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

            length = min(len(ids), args.max_length)
            self.tokens[i][:length] = ids[:length]
            self.mask[i][:length] = 1
            self.h_pos[i] = min(args.max_length-1, ph) 
            self.t_pos[i] = min(args.max_length-1, pt)

            self.h_end[i] = min(args.max_length-1, eh)
            self.t_end[i] = min(args.max_length-1, et)

        print("The number of sentence in which tokenizer can't find head/tail entity is %d" % entityMarker.err)
        # Samples positive pair dynamically. 
        self.__sample__()
    
    def __pos_pair__(self, scope):
        """Generate positive pair.
        Args:
            scope: A scope in which all sentences' label are the same.
                scope example: [0, 12]
        Returns:
            all_pos_pair: All positive pairs. 
            ! IMPORTTANT !
            Given that any sentence pair in scope is positive pair, there
            will be totoally (N-1)N/2 pairs, where N equals scope[1] - scope[0].
            The positive pair's number is proportional to N^2, which will cause 
            instance imbalance. And If we consider all pair, there will be a huge 
            number of positive pairs.
            So we sample positive pair which is proportional to N. And in different epoch,
            we resample sentence pair, i.e. dynamic sampling.
        """
        pos_scope = list(range(scope[0], scope[1]))
        data = json.load(open(os.path.join(self.path, "cpdata.json")))
        
        # shuffle bag to get different pairs
        random.shuffle(pos_scope)
        if self.args.anchor_feature == "one":
            pos_scope.sort(key=lambda x: abs(data[x]['t']['pos'][0][0] - data[x]['h']['pos'][0][0]))
            all_pos_pair = []
            bag = []
            for i, index in enumerate(pos_scope):
                bag.append(index)
                if (i+1) % 2 == 0:
                    all_pos_pair.append(bag)
                    bag = []
            return all_pos_pair
        else:
            all_pos_pair = []
            bag = []
            for i, index in enumerate(pos_scope):
                bag.append(index)
                if (i+1) % 2 == 0:
                    all_pos_pair.append(bag)
                    bag = []
            return all_pos_pair
    
    def __sample__(self):
        """Samples positive pairs.
        After sampling, `self.pos_pair` is all pairs sampled.
        `self.pos_pair` example: 
                [
                    [0,2],
                    [1,6],
                    [12,25],
                    ...
                ]
        """
        data = json.load(open(os.path.join(self.path, "cpdata.json")))
        rel2scope = json.load(open(os.path.join(self.path, "rel2scope.json")))
        self.pos_pair = []
        for rel in rel2scope.keys():
            scope = rel2scope[rel]
            pos_pair = self.__pos_pair__(scope)
            self.pos_pair.extend(pos_pair)

        print("Postive pair's number is %d" % len(self.pos_pair))

    def __len__(self):
        """Number of instances in an epoch.
        
        Overwitten function.
        """
        return len(self.pos_pair)

    def __getitem__(self, index):
        """Get training instance.
        Overwitten function.
        
        Args:
            index: Instance index.
        
        Return:
            input: Tokenized word id.
            mask: Attention mask for bert. 0 means masking, 1 means not masking.
            label: Label for sentence.
            h_pos: Position of head entity.
            t_pos: Position of tail entity.
        """
        bag = self.pos_pair[index]
        input = np.zeros((self.args.max_length * 2), dtype=int)
        mask = np.zeros((self.args.max_length * 2), dtype=int)
        label = np.zeros((2), dtype=int)
        h_pos = np.zeros((2), dtype=int)
        t_pos = np.zeros((2), dtype=int)

        h_end = np.zeros((2), dtype=int)
        t_end = np.zeros((2), dtype=int)



        for i, ind in enumerate(bag):
            input[i*self.args.max_length : (i+1)*self.args.max_length] = self.tokens[ind]
            mask[i*self.args.max_length : (i+1)*self.args.max_length] = self.mask[ind]
            label[i] = self.label[ind]
            h_pos[i] = self.h_pos[ind]
            t_pos[i] = self.t_pos[ind]

            h_end[i] = self.h_end[ind]
            t_end[i] = self.t_end[ind]

        return input, mask, label, h_pos, t_pos, h_end, t_end


class MTBDataset(Dataset):
    """Overwritten class Dataset for model MTB.
    This class prepare data for training of MTB.
    """
    def __init__(self, path, args):
        """Inits tokenized sentence and positive pair for MTB.
        
        Args:
            path: path to your dataset.
            args: args from command line.
        
        Returns:
            No returns
        
        Raises:
            If the dataset in `path` is not the same format as described in 
            file 'prepare_data.py', there may raise:
                - `key nor found`
                - `integer can't be indexed`
                and so on.
        """
        self.path = path 
        self.args = args 
        data = json.load(open(os.path.join(path, "mtbdata.json")))
        entityMarker = EntityMarker()
        
        # Important Configures
        tot_sentence = len(data)

        # Converts tokens to ids and meanwhile `BLANK` some entities randomly.
        self.tokens = np.zeros((tot_sentence, args.max_length), dtype=int)
        self.mask = np.zeros((tot_sentence, args.max_length), dtype=int)
        self.h_pos = np.zeros((tot_sentence), dtype=int)
        self.t_pos = np.zeros((tot_sentence), dtype=int)

        ### revised
        self.h_end = np.zeros((tot_sentence), dtype=int)
        self.t_end = np.zeros((tot_sentence), dtype=int)
        ### revised

        for i, sentence in enumerate(data):
            h_flag = random.random() > args.alpha
            t_flag = random.random() > args.alpha
            h_p = sentence["h"]["pos"][0]  
            t_p = sentence["t"]["pos"][0]
        ### revised
            ids, ph, pt, eh, et = entityMarker.tokenize(sentence["tokens"], [h_p[0], h_p[-1]+1], [t_p[0], t_p[-1]+1], None, None, h_flag, t_flag)
        ### revised
            length = min(len(ids), args.max_length)
            self.tokens[i][0:length] = ids[0:length]
            self.mask[i][0:length] = 1
            self.h_pos[i] = min(args.max_length-1, ph)
            self.t_pos[i] = min(args.max_length-1, pt)
        ### revised
            self.h_end[i] = min(args.max_length-1, eh)
            self.t_end[i] = min(args.max_length-1, et)
        ### revised

        print("The number of sentence in which tokenizer can't find head/tail entity is %d" % entityMarker.err)

        entpair2scope = json.load(open(os.path.join(path, "entpair2scope.json")))
        entpair2negpair = json.load(open(os.path.join(path, "entpair2negpair.json")))
        self.pos_pair = []
        
        # Generates all positive pair.
        for key in entpair2scope.keys():
            self.pos_pair.extend(self.__pos_pair__(entpair2scope[key]))
        print("Positive pairs' number is %d" % len(self.pos_pair))
        # Samples negative pairs dynamically.
        self.__sample__()

    def __sample__(self):    
        """Sample hard negative pairs.
        Sample hard negative pairs for MTB. As described in `prepare_data.py`, 
        `entpair2negpair` is ` A python dict whose key is `head_id#tail_id`. And the value
                is the same format as key, but head_id or tail_id is different(only one id is 
                different). ` 
        
        ! IMPORTANT !
        We firstly get all hard negative pairs which may be a very huge number and then we sam
        ple negaitive pair where sampling number equals positive pairs' numebr. Using our 
        dataset, this code snippet can run normally. But if your own dataset is very big, this 
        code snippet will cost a lot of memory.
        """
        entpair2scope = json.load(open(os.path.join(self.path, "entpair2scope.json")))
        entpair2negpair = json.load(open(os.path.join(self.path, "entpair2negpair.json")))
        neg_pair = []

        # Gets all negative pairs.
        for key in entpair2negpair.keys():
            my_scope = entpair2scope[key]
            entpairs = entpair2negpair[key]
            if len(entpairs) == 0:
                continue
            for entpair in entpairs:
                neg_scope = entpair2scope[entpair]
                neg_pair.extend(self.__neg_pair__(my_scope, neg_scope))
        print("(MTB)Negative pairs number is %d" %len(neg_pair))
        
        # Samples a same number of negative pair with positive pairs. 
        random.shuffle(neg_pair)
        self.neg_pair = neg_pair[0:len(self.pos_pair)]
        del neg_pair # save the memory 

    def __pos_pair__(self, scope):
        """Gets all positive pairs.
        Args:
            scope: A scope in which all sentences share the same
                entity pair(head entity and tail entity).
        
        Returns:
            pos_pair: All positive pairs in a scope. The number of 
                positive pairs in a scope is (N-1)N/2 where N equals
                scope[1] - scope[0]
        """
        ent_scope = list(range(scope[0], scope[1]))
        pos_pair = []
        for i in range(len(ent_scope)):
            for j in range(i+1, len(ent_scope)):
                pos_pair.append([ent_scope[i], ent_scope[j]])
        return pos_pair

    def __neg_pair__(self, my_scope, neg_scope):
        """Gets all negative pairs in different scope.
        Args:
            my_scope: A scope which is samling negative pairs.
            neg_scope: A scope where sentences share only one entity
                with sentences in my_scope.
        
        Returns:
            neg_pair: All negative pair. Sentences in different scope 
                make up negative pairs.
        """
        my_scope = list(range(my_scope[0], my_scope[1]))
        neg_scope = list(range(neg_scope[0], neg_scope[1]))
        neg_pair = []
        for i in my_scope:
            for j in neg_scope:
                neg_pair.append([i, j])
        return neg_pair

    def __len__(self):
        """Number of instances in an epoch.
        
        Overwitten function.
        """
        return len(self.pos_pair)

    def __getitem__(self, index):
        """Gets training instance.
        If index is odd, we will return nagative instance, otherwise 
        positive instance. So in a batch, the number of positive pairs 
        equal the number of negative pairs.
        Args:
            index: Data index.
        
        Returns:
            {l,h}_input: Tokenized word id.
            {l,h}_mask: Attention mask for bert. 0 means masking, 1 means not masking.
            {l,h}_ph: Position of head entity.
            {l,h}_pt: Position of tail entity.
            label: Positive or negative.
            Setences in the same position in l_input and r_input is a sentence pair
            (positive or negative).
        """
        if index % 2 == 0:
            l_ind = self.pos_pair[index][0]
            r_ind = self.pos_pair[index][1]
            label = 1
        else:
            l_ind = self.neg_pair[index][0]
            r_ind = self.neg_pair[index][1]
            label = 0
        
        l_input = self.tokens[l_ind]
        l_mask = self.mask[l_ind]
        l_ph = self.h_pos[l_ind]
        l_pt = self.t_pos[l_ind]
        ### revised
        l_eh = self.h_end[l_ind]
        l_et = self.t_end[l_ind]
        ### revised
        r_input = self.tokens[r_ind]
        r_mask = self.mask[r_ind]
        r_ph = self.h_pos[r_ind]
        r_pt = self.t_pos[r_ind]
        ### revised
        r_eh = self.h_end[r_ind]
        r_et = self.t_end[r_ind]
        ### revised
        ### revised
        return l_input, l_mask, l_ph, l_pt, r_input, r_mask, r_ph, r_pt, label, l_eh, l_et, r_eh, r_et
        ### revised