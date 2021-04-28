import os
import pdb
import copy
import torch
import torch.nn as nn
from pytorch_metric_learning.losses import NTXentLoss
from transformers import BertForMaskedLM, BertForPreTraining, BertTokenizer
from sentence_transformers import SentenceTransformer

def mask_tokens(inputs, tokenizer, not_mask_pos=None):
    """ Prepare masked tokens of inputs and labels for masked language modeling (80% MASK, 10% random, 10% original).

    Args:
        inputs: Inputs to mask. [batch_size, max_length]
        tokenizer: Tokenizer.
        not_mask_pos: Used to forbid masking entity mentions. 1 for not mask else 0.
    Returns:
        inputs: Masked input tokens.
        labels: Masked language model label tokens.
    """
    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. \
                Remove the --mlm flag if you want to use this tokenizer."
        )

    labels = copy.deepcopy(inputs)

    # sample a few tokens in each sequence for masked LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, 0.15)
    special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)

    if tokenizer.pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    
    if not_mask_pos is None:
        masked_indices = torch.bernoulli(probability_matrix).bool()
    else:
        masked_indices = torch.bernoulli(probability_matrix).bool() & (~(not_mask_pos.bool()))
    
    labels[~masked_indices] = -100  # only compute loss on masked tokens

    # replace masked input tokens with tokenizer.mask_token ([MASK]) - 80% probability
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # replace masked input tokens with random word - 10%
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    return inputs.cuda(), labels.cuda()

class TS_CP_SBERT(nn.Module):
    """ Teacher-Student model between CP model and SBERT model.
    Uses MSELoss.

    Attributes:
        teacher_model: Model used as a teacher model.
        student_model: Model to train.
        tokenizer: Tokenizer.
        mseloss: MSELoss.
        args: args from command line.
    """
    def __init__(self, args):
        super().__init__()
        # teacher model
        teacher_model_ckpt = torch.load(args.teacher_model)
        self.teacher_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        model_dict = self.teacher_model.state_dict()
        pretrained_dict = {k: v for k, v in teacher_model_ckpt['bert-base'].items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.teacher_model.load_state_dict(model_dict)

        # student model
        self.student_model = SentenceTransformer('bert-base-nli-mean-tokens')

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.mseloss = nn.MSELoss()
        self.args = args

    def forward(self, input, mask, label, h_pos, t_pos, h_end, t_end, raw_text_id):
        # input shape = [batch size, max len]
        entity_marker_text = []
        raw_text = []
        batch_size = input.size()[0]
        for i in range(batch_size):
            entity_marker_text.append(self.tokenizer.decode(input[i]))
            raw_text.append(self.tokenizer.decode(raw_text_id[i]))

        indice = torch.arange(0, batch_size)
        not_mask_pos = torch.zeros((input.size()[0], input.size()[1]), dtype=int)
        not_mask_pos[indice, h_pos] = 1
        not_mask_pos[indice, t_pos] = 1
        not_mask_pos[indice, h_end] = 1
        not_mask_pos[indice, t_end] = 1

        m_input, m_labels = mask_tokens(input.cpu(), self.tokenizer, not_mask_pos=not_mask_pos)
        teacher_outputs = self.teacher_model(input_ids=m_input, labels=m_labels, attention_mask=mask, output_hidden_states=True)
        student_outputs_raw_text = self.student_model.encode(raw_text, convert_to_tensor=True)   # [batch size, hidden size]
        student_outputs_entity_marker_text = self.student_model.encode(entity_marker_text, convert_to_tensor=True)   # [batch size, hidden size]

        student_state_1 = student_outputs_raw_text
        student_state_2 = student_outputs_entity_marker_text

        teacher_loss = teacher_outputs.loss
        teacher_logits = teacher_outputs.logits
        teacher_last_hidden_state = teacher_outputs.hidden_states[-1]
        
        teacher_h_start_state = teacher_last_hidden_state[indice, h_pos]      # [batch size, hidden size]
        teacher_h_end_state = teacher_last_hidden_state[indice, h_end]
        teacher_t_start_state = teacher_last_hidden_state[indice, t_pos]
        teacher_t_end_state = teacher_last_hidden_state[indice, t_end]

        teacher_state = torch.cat([teacher_h_start_state, teacher_t_start_state], 1)      # [batch size, hidden size * 2]

        if self.args.pooling_method == "mean":
            teacher_state = teacher_state.view(-1, 2)  # [batch size * hidden size, 2]
            teacher_state = teacher_state.mean(dim=1).view(batch_size, -1)    # [batch size, hidden size]
        elif self.args.pooling_method == "max":
            teacher_state = teacher_state.view(-1, 2)   # [batch size * hidden size, 2]
            teacher_state = torch.max(teacher_state, dim=1)[0].view(batch_size, -1)   # [batch size, hidden size]
        elif self.args.pooling_method == "min":
            teacher_state = teacher_state.view(-1, 2)   # [batch size * hidden size, 2]
            teacher_state = torch.min(teacher_state, dim=1)[0].view(batch_size, -1)   # [batch size, hidden size]

        t_s_loss = self.mseloss(teacher_state, student_state_1)
        s_s_loss = self.mseloss(student_state_1, student_state_2)

        return t_s_loss, s_s_loss

class TS(nn.Module):
    """ Teacher-Student model.

    This class implements 'TS' model based on model 'BertForMaskedLM' and 'CP'.
    Uses MSELoss.

    Attributes:
        teacher_model: Model used as a teacher model.
        student_model: Model to train.
        tokenizer: Tokenizer.
        mseloss: MSELoss.
        args: args from command line.
    """
    def __init__(self, args):
        super().__init__()
        teacher_model_ckpt = torch.load(args.teacher_model)
        self.teacher_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        model_dict = self.teacher_model.state_dict()
        pretrained_dict = {k: v for k, v in teacher_model_ckpt['bert-base'].items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.teacher_model.load_state_dict(model_dict)

        self.student_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.mseloss = nn.MSELoss()
        self.args = args

    def forward(self, input, mask, label, h_pos, t_pos, h_end, t_end):
        input = input.view(-1, self.args.max_length)     # [batch size, max len * 2] -> [batch size * 2, max len]
        mask = mask.view(-1, self.args.max_length)
        label = label.view(-1)     # [batch size * 2]
        h_pos = h_pos.view(-1)
        t_pos = t_pos.view(-1)
        h_end = h_end.view(-1)
        t_end = t_end.view(-1)

        indice = torch.arange(0, input.size()[0])
        not_mask_pos = torch.zeros((input.size()[0], input.size()[1]), dtype=int)
        not_mask_pos[indice, h_pos] = 1
        not_mask_pos[indice, t_pos] = 1
        not_mask_pos[indice, h_end] = 1
        not_mask_pos[indice, t_end] = 1

        m_input, m_labels = mask_tokens(input.cpu(), self.tokenizer, not_mask_pos=not_mask_pos)
        teacher_outputs = self.teacher_model(input_ids=m_input, labels=m_labels, attention_mask=mask, output_hidden_states=True)
        student_outputs = self.student_model(input_ids=m_input, labels=m_labels, attention_mask=mask, output_hidden_states=True)
        
        teacher_loss = teacher_outputs.loss
        teacher_logits = teacher_outputs.logits
        teacher_last_hidden_state = teacher_outputs.hidden_states[-1]
        student_loss = student_outputs.loss
        student_logits = student_outputs.logits
        student_last_hidden_state = student_outputs.hidden_states[-1]

        teacher_h_start_state = teacher_last_hidden_state[indice, h_pos]
        teacher_h_end_state = teacher_last_hidden_state[indice, h_end]
        teacher_t_start_state = teacher_last_hidden_state[indice, t_pos]
        teacher_t_end_state = teacher_last_hidden_state[indice, t_end]
        
        student_h_start_state = student_last_hidden_state[indice, h_pos]
        student_h_end_state = student_last_hidden_state[indice, h_end]
        student_t_start_state = student_last_hidden_state[indice, t_pos]
        student_t_end_state = student_last_hidden_state[indice, t_end]

        if self.args.output_representation == "entity_marker":
            teacher_state = torch.cat([teacher_h_start_state, teacher_t_start_state], 1)
            student_state = torch.cat([student_h_start_state, student_t_start_state], 1)

            teacher_state = teacher_state.view(-1, self.args.hidden_size * 4)         # [batch size, hidden size * 4]
            teacher_state = teacher_state[:, 0:self.args.hidden_size * 2]              # [batch size, hidden size * 2]
            student_state = student_state.view(-1, self.args.hidden_size * 4)
            student_state_1 = student_state[:, 0:self.args.hidden_size * 2]
            student_state_2 = student_state[:, self.args.hidden_size * 2:]
            
        elif self.args.output_representation == "end_to_first_concat":
            teacher_state = torch.cat([teacher_h_end_state, teacher_t_start_state], 1)          # [batch size * 2, hidden_size * 2]
            student_state = torch.cat([student_h_end_state, student_t_start_state], 1)          # [batch size * 2, hidden_size * 2]

            teacher_state = teacher_state.view(-1, self.args.hidden_size * 4)         # [batch size, hidden size * 4]
            teacher_state = teacher_state[:, 0:self.args.hidden_size * 2]              # [batch size, hidden size * 2]
            student_state = student_state.view(-1, self.args.hidden_size * 4)
            student_state_1 = student_state[:, 0:self.args.hidden_size * 2]
            student_state_2 = student_state[:, self.args.hidden_size * 2:]

        t_s_loss = self.mseloss(teacher_state, student_state_1)
        s_s_loss = self.mseloss(student_state_1, student_state_2)

        # return student_loss, t_s_loss + s_s_loss
        return t_s_loss, s_s_loss

class CP(nn.Module):
    """ Contrastive Pre-training model.

    This class implements 'CP' model based on model 'BertForMaskedLM'.
    Uses NTXentLoss as constrastive loss function.

    Attributes:
        model: Model to train.
        tokenizer: Tokenizer.
        ntxloss: Contrastive loss function.
        args: args from command line.
    """
    def __init__(self, args):
        super().__init__()
        self.model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.ntxloss = NTXentLoss(temperature=args.temperature)
        self.args = args
        ######################## revised ###########################
    def forward(self, input, mask, label, h_pos, t_pos, h_end, t_end):
        ######################## revised ###########################
        input = input.view(-1, self.args.max_length)
        mask = mask.view(-1, self.args.max_length)
        label = label.view(-1)  # [batch_size * 2]
        h_pos = h_pos.view(-1)
        t_pos = t_pos.view(-1)
        ######################## revised ###########################
        h_end = h_end.view(-1)
        t_end = t_end.view(-1)
        ######################## revised ###########################


        # mask_tokens function does not mask entity mention.
        indice = torch.arange(0, input.size()[0])
        not_mask_pos = torch.zeros((input.size()[0], input.size()[1]), dtype=int)
        not_mask_pos[indice, h_pos] = 1
        not_mask_pos[indice, t_pos] = 1
        ######################## revised ###########################
        not_mask_pos[indice, h_end] = 1
        not_mask_pos[indice, t_end] = 1
        ######################## revised ###########################

        m_input, m_labels = mask_tokens(input.cpu(), self.tokenizer, not_mask_pos=not_mask_pos)
        m_outputs = self.model(input_ids=m_input, labels=m_labels, attention_mask=mask, output_hidden_states=True)
        m_loss = m_outputs.loss
        m_logits = m_outputs.logits
        m_last_hidden_state = m_outputs.hidden_states[-1]
        
        ##### revised
        # # entity marker starter
        # batch_size = input.size()[0]
        # h_state = m_last_hidden_state[indice, h_pos]  # [batch_size * 2, hidden_size]
        # t_state = m_last_hidden_state[indice, t_pos]
        # state = torch.cat((h_state, t_state), 1)

        h_start_state = m_last_hidden_state[indice, h_pos]
        h_end_state = m_last_hidden_state[indice, h_end]
        t_start_state = m_last_hidden_state[indice, t_pos]
        t_end_state = m_last_hidden_state[indice, t_end]

        if self.args.output_representation == "entity_marker":
            state = torch.cat((h_start_state, t_start_state), 1)
        elif self.args.output_representation == "all_markers":
            h_start_state = h_start_state.unsqueeze(2)
            h_end_state = h_end_state.unsqueeze(2)
            t_start_state = t_start_state.unsqueeze(2)
            t_end_state = t_end_state.unsqueeze(2)
            state = torch.cat([h_start_state, h_end_state, t_start_state, t_end_state], 2)
            state = torch.max(state, dim=2)[0]
        elif self.args.output_representation == "end_to_first":
            h_end_state = h_end_state.unsqueeze(2)
            t_start_state = t_start_state.unsqueeze(2)
            state = torch.cat([h_end_state, t_start_state], 2)
            state = torch.max(state, dim=2)[0]
        elif self.args.output_representation == "all_markers_concat":
            state = torch.cat([h_start_state, h_end_state, t_start_state, t_end_state], 1)
        elif self.args.output_representation == "end_to_first_concat":
            state = torch.cat([h_end_state, t_start_state], 1)
        elif self.args.output_representation == "marker_minus":
            state = t_start_state - h_start_state
        else:   # CLS
            state = m_last_hidden_state[:, 0, :]

        ##### revised
        r_loss = self.ntxloss(state, label)

        return m_loss, r_loss
        
class MTB(nn.Module):
    """ Matching the Blanks.

    This class implements 'MTB' model based on model 'BertForMaskedLM'.

    Attributes:
        model: Model to train.
        tokenizer: Tokenizer.
        bceloss: Binary Cross Entropy loss.
    """
    def __init__(self, args):
        super().__init__()
        self.model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bceloss = nn.BCEWithLogitsLoss()
        self.args = args
    
    def forward(self, l_input, l_mask, l_ph, l_pt, r_input, r_mask, r_ph, r_pt, label):
        # compute not mask entity marker
        indice = torch.arange(0, l_input.size()[0])
        l_not_mask_pos = torch.zeros((l_input.size()[0], l_input.size()[1]), dtype=int)
        r_not_mask_pos = torch.zeros((l_input.size()[0], l_input.size()[1]), dtype=int)

        # mask_tokens function does not mask entity mention.
        l_not_mask_pos[indice, l_ph] = 1
        l_not_mask_pos[indice, l_pt] = 1
        r_not_mask_pos[indice, r_ph] = 1
        r_not_mask_pos[indice, r_pt] = 1

        m_l_input, m_l_labels = mask_tokens(l_input.cpu(), self.tokenizer, l_not_mask_pos)
        m_r_input, m_r_labels = mask_tokens(r_input.cpu(), self.tokenizer, r_not_mask_pos)
        m_l_outputs = self.model(input_ids=m_l_input, labels=m_l_labels, attention_mask=l_mask)
        m_r_outputs = self.model(input_ids=m_r_input, labels=m_r_labels, attention_mask=r_mask)
        
        m_loss = m_l_outputs.loss + m_r_outputs.loss
        m_l_logits = m_l_outputs.logits
        m_r_logits = m_r_outputs.logits

        batch_size = l_input.size()[0]
        
        # left output
        l_h_state = m_l_logits[indice, l_ph]  # [batch, hidden_size]
        l_t_state = m_l_logits[indice, l_pt]
        l_state = torch.cat((l_h_state, l_t_state), 1)  # [batch, hidden_size * 2]
        
        # right output
        r_h_state = m_r_logits[indice, r_ph]  # [batch, hidden_size]
        r_t_state = m_r_logits[indice, r_pt]
        r_state = torch.cat((r_h_state, r_t_state), 1)  # [batch, hidden_size * 2]

        # calculate similarity
        similarity = torch.sum(l_state * r_state, 1)

        # calculate loss
        r_loss = self.bceloss(similarity, label.float())

        return m_loss, r_loss