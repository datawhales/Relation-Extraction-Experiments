import torch
import pdb
import torch.nn as nn
from transformers import BertModel

class REModel(nn.Module):
    """ Relation Extraction Model.
    """
    def __init__(self, args, weight=None):
        super().__init__()
        self.args = args
        self.training = True

        if weight is None:
            self.loss = nn.CrossEntropyLoss()
        else:
            print("CrossEntropy Loss has weight!")
            self.loss = nn.CrossEntropyLoss(weight=weight)

        #################### modified ######################
        # scale = 2 if args.output_representation == "entity_marker" else 1
        ## all_markers_concat -> scale 4
        ## entity_marker, end_to_first_concat -> scale 2
        ## all_markers, end_to_first, [CLS] -> scale 1
        if args.output_representation == "all_markers_concat":
            scale = 4
        elif args.output_representation == "entity_marker" or args.output_representation == "end_to_first_concat":
            scale = 2
        else:
            scale = 1
        #################### modified ######################
        self.rel_fc = nn.Linear(args.hidden_size * scale, args.rel_num)
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        if args.ckpt_to_load != "None":
            print("******** load from ckpt/" + args.ckpt_to_load + " ********")
            ckpt = torch.load("../../../pretrain/ckpt/" + args.ckpt_to_load)
            model_dict = self.bert.state_dict()
            pretrained_dict = {k: v for k, v in ckpt['bert-base'].items() if k in model_dict}

            model_dict.update(pretrained_dict)
            self.bert.load_state_dict(model_dict)
        else:
            print("******** No ckpt to load, we will use bert base! ********")
        #################### modified ######################
    def forward(self, input_ids, mask, h_pos, t_pos, label, h_end, t_end):
        #################### modified ######################

        # bert encode
        outputs = self.bert(input_ids, mask)

        last_hidden_states = outputs.last_hidden_state    # [batch size, sequence length, hidden size]
        pooler_output = outputs.pooler_output        # [batch size, hidden size]

        # entity marker
        if self.args.output_representation == "entity_marker":
            indice = torch.arange(input_ids.size()[0])    # batch size
            h_state = last_hidden_states[indice, h_pos]
            t_state = last_hidden_states[indice, t_pos]
            state = torch.cat((h_state, t_state), 1)    # [batch size, hidden size * 2]
        elif self.args.output_representation == "all_markers":
            indice = torch.arange(input_ids.size()[0])    # batch size
            h_start_state = last_hidden_states[indice, h_pos]
            h_end_state = last_hidden_states[indice, h_end]
            t_start_state = last_hidden_states[indice, t_pos]
            t_end_state = last_hidden_states[indice, t_end]
            h_start_state = h_start_state.unsqueeze(2)
            h_end_state = h_end_state.unsqueeze(2)
            t_start_state = t_start_state.unsqueeze(2)
            t_end_state = t_end_state.unsqueeze(2)
            state = torch.cat([h_start_state, h_end_state, t_start_state, t_end_state], 2)
            state = torch.max(state, dim=2)[0]
        elif self.args.output_representation == "end_to_first":
            indice = torch.arange(input_ids.size()[0])    # batch size
            h_end_state = last_hidden_states[indice, h_end]
            t_start_state = last_hidden_states[indice, t_pos]
            h_end_state = h_end_state.unsqueeze(2)
            t_start_state = t_start_state.unsqueeze(2)
            state = torch.cat([h_end_state, t_start_state], 2)
            state = torch.max(state, dim=2)[0]
        elif self.args.output_representation == "all_markers_concat":
            indice = torch.arange(input_ids.size()[0])    # batch size
            h_start_state = m_last_hidden_state[indice, h_pos]
            h_end_state = m_last_hidden_state[indice, h_end]
            t_start_state = m_last_hidden_state[indice, t_pos]
            t_end_state = m_last_hidden_state[indice, t_end]
            state = torch.cat([h_start_state, h_end_state, t_start_state, t_end_state], 1)
        elif self.args.output_representation == "end_to_first_concat":
            indice = torch.arange(input_ids.size()[0])    # batch size
            h_end_state = m_last_hidden_state[indice, h_end]
            t_start_state = m_last_hidden_state[indice, t_pos]
            state = torch.cat([h_end_state, t_start_state], 1)    
        else:  # [CLS]
            state = last_hidden_states[:, 0, :]   # [batch size, hidden size]
        
        # linear map
        logits = self.rel_fc(state)   # [batch size, rel num]
        _, output = torch.max(logits, 1)

        if self.training:
            loss = self.loss(logits, label)
            return loss, output
        else:
            return logits, output