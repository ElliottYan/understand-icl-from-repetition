import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json
import tqdm

os.environ['HOME'] = "./"

# sentence list from DITTO repo
sentence_list = {
    0:{
        'base_sentence': 'His mother was a college softball player for the NEO Lady Norse .',
        'keyword': 'super'
    },
    1:{
        'base_sentence': 'Many courts have declined to assess the religious status of Scientology .',
        'keyword': 'religious'
    },
    2: {
        'base_sentence': 'I like to play basketball .',
        'keyword': 'like'
    },
   3: {
        'base_sentence': 'In a post to his blog , Blythe explained that he met the family in private after the trial .',
        'keyword': '\.'
    },
}

def build_common_parser(args=None):
    parser = argparse.ArgumentParser(
        description="Analyzing scripts for self-reinforcement effect.",
        usage="python analyze_self_reinforce.py [<args>] [-h | --help]"
    )

    # input files
    parser.add_argument("--model_family", type=str, default='llama')
    parser.add_argument("--model_max_size", type=str, default='70b')
    parser.add_argument("--model_min_size", type=str, default='125m')
    parser.add_argument("--repeat_num", type=int, default=101)
    parser.add_argument("--output_dir", type=str, default="")

    return parser

def load_model_and_tokenizer(model_path):
#     device = 'cuda'
    if "llama" not in model_path:
        model = AutoModelForCausalLM.from_pretrained(model_path,device_map = 'auto', torch_dtype=torch.float16, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        from transformers import LlamaForCausalLM, LlamaTokenizer
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map = 'auto', torch_dtype=torch.float16, trust_remote_code=True)
        model.tie_weights()
        
        # Warning: should use a correct version of tokenizer of llama
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
        # # process for llama
        # DEFAULT_PAD_TOKEN = "[PAD]"
        # DEFAULT_EOS_TOKEN = "</s>"
        # DEFAULT_BOS_TOKEN = "<s>"
        # DEFAULT_UNK_TOKEN = "<unk>"
        # special_tokens_dict = dict()
        # if tokenizer.pad_token is None:
        #     special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
        # if tokenizer.eos_token is None:
        #     special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
        # if tokenizer.bos_token is None:
        #     special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
        # if tokenizer.unk_token is None:
        #     special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
        # tokenizer.add_special_tokens(
        #             {
        #                 "eos_token": DEFAULT_EOS_TOKEN,
        #                 "bos_token": DEFAULT_BOS_TOKEN,
        #                 "unk_token": DEFAULT_UNK_TOKEN,
        #             })
        
    model.eval()
#     model.to(device)

    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def shift_left(x, pad=1):
    """
    Shifts a tensor left by one step.

    Args:
        x: A PyTorch tensor of shape (batch_size, sequence_length, hidden_size).

    Returns:
        A PyTorch tensor of the same shape as x.
    """
    # Get the batch size, sequence length, and hidden size of the tensor
    batch_size, sequence_length = x.shape[:2]

    # Create a zero tensor with the same shape as x
    shifted_x = torch.zeros_like(x).fill_(pad)

    # Shift each row of the tensor left by one step
    shifted_x[:, :-1] = x[:, 1:]

    return shifted_x

def get_mask(x, pad):
    mask = (x == pad)
    return mask

def m_mean(x, mask, dim):
    assert x.shape == mask.shape
    ret = (x*mask).sum(dim) / mask.sum(dim)
    return ret

def custom_gather(value_tensor, targets, pad, dim=-1):
    '''
    Case (A): value_tensor: [bsz, n, l, V], targets: [bsz, l]
    Case (B): value_tensor: [bsz, n, l, V], targets: [bsz, l, K]
    Case (C): value_tensor: [bsz, L, V], targets: [bsz, L]
    Case (D): value_tensor: [bsz, L, V], targets: [bsz, L, K]
    '''
    if value_tensor.ndim == 4:
        # Case (A) and (B)
        assert targets.ndim in [2, 3]
        assert targets.shape[0] == value_tensor.shape[0] and targets.shape[1] == value_tensor.shape[2]
        if targets.ndim == 2:
            # targets: [bsz, l], value_tensor: [bsz, n_iter, l, V]
            index = targets[:, None, :,None].expand(value_tensor.shape[0], value_tensor.shape[1], value_tensor.shape[2], -1).clone()
        else:
            # targets: [bsz, l, K], value_tensor: [bsz, n_iter, l, V]
            index = targets[:, None, :].expand(value_tensor.shape[0], value_tensor.shape[1], value_tensor.shape[2], -1).clone()
        gathered = torch.gather(value_tensor, dim=-1, index=index)
        mask = (index != pad)
        return gathered, mask
    elif value_tensor.ndim == 3:
        # Case (C) or (D)
        assert targets.ndim in [2, 3]
        assert targets.shape[0] == value_tensor.shape[0] and targets.shape[1] == value_tensor.shape[1]
        if targets.ndim == 2:
            index = targets[:, :, None] # [bsz, L, 1]
        else:
            index = targets # [bsz, L, K]
        
        gathered = torch.gather(value_tensor, 2, index) # [bsz, L, K]
        mask = (index != pad)
        return gathered, mask
    else:
        raise ValueError()
        
def compute_TP(x_dict, pad=1):
    if x_dict["target_tokens"] is not None:
        probs = x_dict['split_probs']
        targets = x_dict['target_tokens'] # [bsz, l]
        gathered, mask = custom_gather(probs, targets, pad)
        gathered = gathered.sum(-1) # [bsz, N, L, K] -> [bsz, N, L]
        TP = m_mean(gathered, mask, 2).mean(0) # [N, ]
    elif "full_targets" in x_dict and x_dict["full_targets"] is not None:
        probs = x_dict['full_probs']
        targets = x_dict['full_targets']
        index_mask = x_dict['full_index_mask'] # [bsz, L]
        gathered, mask = custom_gather(probs, targets, pad=pad)
        gathered = gathered.sum(-1) # [bsz, L, K] -> [bsz, L]
        TP = []
        for i in range(index_mask.max().item()):
            cur_mask = (index_mask == i+1)
            TP.append(m_mean(gathered, cur_mask, dim=1))
        TP = torch.cat(TP, dim=0) # [N, ]
        pass
    else:
        raise ValueError()
    return TP

def compute_TP_over_pattern(x_dict, pad=1):
    probs = x_dict['full_probs']
    targets = x_dict['full_targets']
    index_mask = x_dict['pattern_index_mask'] # [bsz, L]
    gathered, mask = custom_gather(probs, targets, pad=pad)
    gathered = gathered.sum(-1) # [bsz, L, K] -> [bsz, L]
    TP = []
    for i in range(index_mask.max().item()):
        cur_mask = (index_mask == i+1)
        TP.append(m_mean(gathered, cur_mask, dim=1))
    TP = torch.cat(TP, dim=0) # [N, ]
    return TP


def compute_TP_except_targets(x_dict, pad=1):
    pass
    input_tokens = x_dict['input_tokens']
    shifted_input = shift_left(input_tokens, pad=pad) # []

# compute TP of the first token.
def compute_TP_first(x_dict, pad=1):
    if x_dict["target_tokens"] is not None:
        probs = x_dict['split_probs']
        targets = x_dict['target_tokens'].clone() # [bsz, l]
        # this code can only work with bsz=1
        assert targets.shape[0] == 1
        # NOTE: bug here!!!!
        valid_mask = (targets != pad).int()
        nonzeros = valid_mask.nonzero(as_tuple=True)
        exclude_first_nonzeros = tuple([item[1:] for item in nonzeros])
        # set these to zeros
        targets[exclude_first_nonzeros] = pad
        gathered, mask = custom_gather(probs, targets, pad)
        TP = m_mean(gathered, mask, 2).mean(0).sum(-1)
    elif "full_targets" in x_dict and x_dict["full_targets"] is not None:
        probs = x_dict['full_probs']
        targets = x_dict['full_targets']
        index_mask = x_dict['full_index_mask'] # [1, L]
        # N = index_mask.max()[0].item()
        gathered, mask = custom_gather(probs, targets, pad=pad)
        
        gathered = gathered.sum(-1) # [bsz, L]
        
        nonzeros = (index_mask>0).nonzero(as_tuple=True)
        cur = 0
        TP = []
        for i in range(len(nonzeros[0])):
            tup = [item[i] for item in nonzeros]
            if index_mask[tup].item() > cur:
                cur = index_mask[tup]
                TP.append(gathered[tup])
            else:
                continue
        TP = torch.stack(TP, dim=0)
    else:
        raise ValueError()
        
    return TP

def compute_TP_except_first(x_dict, pad=1):
    if x_dict["target_tokens"] is not None:
        raise NotImplementedError()
    elif "full_targets" in x_dict and x_dict["full_targets"] is not None:
        probs = x_dict['full_probs']
        targets = x_dict['full_targets']
        index_mask = x_dict['full_index_mask'] # [1, L]
        # N = index_mask.max()[0].item()
        gathered, mask = custom_gather(probs, targets, pad=pad)
        
        gathered = gathered.sum(-1) # [bsz, L]
        
        nonzeros = (index_mask>0).nonzero(as_tuple=True)
        cur = 0
        TP = []
        for i in range(len(nonzeros[0])):
            tup = [item[i] for item in nonzeros]
            if index_mask[tup].item() > cur:
                cur = index_mask[tup]
                cur_mask = (index_mask == cur)
                cur_mask[tup] = False
                TP.append(gathered[cur_mask].mean())
            else:
                continue
        TP = torch.stack(TP, dim=0)
    else:
        raise ValueError()
        
    return TP

# compute TP of the last token.
def compute_TP_last(x_dict, pad=1):
    if x_dict["target_tokens"] is not None:
        probs = x_dict['split_probs']
        targets = x_dict['target_tokens'].clone() # [bsz, l]
        # this code can only work with bsz=1
        assert targets.shape[0] == 1
        # nonzeros = targets[0, :].nonzero()
        nonzeros = targets.nonzero(as_tuple=True)
        # targets[nonzeros[1:]]
        exclude_last_nonzeros = tuple([item[:-1] for item in nonzeros])
        # set these to zeros
        targets[exclude_last_nonzeros] = pad
        # gather probs
        gathered, mask = custom_gather(probs, targets, pad)
        TP = m_mean(gathered, mask, 2).mean(0).sum(-1)
    elif "full_targets" in x_dict and x_dict["full_targets"] is not None:
        probs = x_dict['full_probs']
        targets = x_dict['full_targets']
        index_mask = x_dict['full_index_mask'] # [1, L]
        gathered, mask = custom_gather(probs, targets, pad=pad)
        
        gathered = gathered.sum(-1) # [bsz, L]
         
        nonzeros = (index_mask>0).nonzero(as_tuple=True)
        N = index_mask.max().item()
        cur = N+1
        TP = []
        # reverse 
        for i in range(len(nonzeros[0])-1, -1, -1):
            tup = [item[i] for item in nonzeros]
            if index_mask[tup].item() < cur:
                cur = index_mask[tup]
                TP.append(gathered[tup])
            else:
                continue

        TP = TP[::-1]
        TP = torch.stack(TP)
    else:
        raise ValueError()
        
    # assert TP.isnan().any().item() is False
    return TP

# comparison
def compute_IP(x_dict, pad=1):
    # return dummy tensor
    # if x_dict["target_tokens"] is None:
        # return torch.tensor([0.,])
    if x_dict["target_tokens"] is not None:
        probs = x_dict['split_probs']
        targets = x_dict['target_tokens'] # [bsz, l]
        gathered, mask = custom_gather(probs, targets, pad)
        IP = m_mean((gathered[:, :1] < gathered[:, 1:]).float(), mask[:, 1:], 2).mean(0).sum(-1)
    elif "full_targets" in x_dict and x_dict["full_targets"] is not None:
        # import pdb; pdb.set_trace()
        probs = x_dict['full_probs']
        targets = x_dict['full_targets']
        index_mask = x_dict['full_index_mask'] # [1, L]
        gathered, mask = custom_gather(probs, targets, pad=pad)
        
        gathered = gathered.sum(-1) # [bsz, L]
         
        # nonzeros = (index_mask>0).nonzero(as_tuple=True)
        N = index_mask.max().item()
        # cur = N+1
        IP = []
        baseline_nonzeros = (index_mask == 1).nonzero(as_tuple=True)
        baseline_probs = gathered[baseline_nonzeros].sum().item()
        
        for i in range(1, N+1):
            cur_nonzeros = (index_mask == i).nonzero(as_tuple=True)
            cur_probs = gathered[cur_nonzeros].sum().item()
            if cur_probs > baseline_probs:
                IP.append(1)
            else:
                IP.append(0)

        IP = torch.tensor(IP) #[N, ]

    return IP

def compute_WR(x_dict, pad=1):
    # return dummy tensor
    if x_dict["target_tokens"] is None:
        return torch.tensor([0.,])
    # winner rate
    probs = x_dict['split_probs']
    targets = x_dict['target_tokens'] # [bsz, l]
    # WR for several targets is yet properly defined.
    # return dummy results
    if targets.ndim == 3:
        return torch.zeros([1, ])
    # compute IP
    index = targets[:, None, :,None].expand(probs.shape[0], probs.shape[1], probs.shape[2], 1)
    mask = (index != pad)
    gathered = torch.gather(probs, dim=-1, index=index).squeeze(-1) # [bsz, n, l]
    IP = (gathered[:, :1] < gathered[:, 1:]).float()
    
    max_prob = probs.max(-1)[0] # [bsz, n, l]
    TEM = (max_prob == gathered)[:, 1:].float()
    WR = (IP * TEM)
    avg_WR = m_mean(WR, mask[:, 1:].squeeze(-1), 2).mean(0)
    return avg_WR

class Stats:
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def add_data(self, x):
        assert x != float('nan')
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2
        assert self.mean != float('nan')
        assert self.M2 != float('nan')

    def get_mean(self):
        return self.mean

    def get_variance(self):
        if self.n < 2:
            return float('nan')
        else:
            return self.M2 / (self.n - 1)


def gather_and_update_all_metrics(stats_dict, ret_dict, tokenizer):
    TP_tensor = compute_TP(ret_dict, pad=tokenizer.pad_token_id)
    TPF_tensor = compute_TP_first(ret_dict, pad=tokenizer.pad_token_id)
    TPEF_tensor = compute_TP_except_first(ret_dict, pad=tokenizer.pad_token_id)
    TPL_tensor = compute_TP_last(ret_dict, pad=tokenizer.pad_token_id)
    IP_tensor = compute_IP(ret_dict, pad=tokenizer.pad_token_id)
    WR_tensor = compute_WR(ret_dict, pad=tokenizer.pad_token_id)

    from collections import defaultdict

    keys = ["TP", "IP", "WR", "TPF", "TPL", "TPEF"]
    tensors = [TP_tensor, IP_tensor, WR_tensor, TPF_tensor, TPL_tensor, TPEF_tensor]
    if "pattern_index_mask" in ret_dict:
        keys.append("TPPA")
        TPPA = compute_TP_over_pattern(ret_dict, pad=tokenizer.pad_token_id)
        tensors.append(TPPA)

    for i in range(len(keys)):
        # init
        key = keys[i]
        t = tensors[i]
        # skip if tensor has nan
        if t.isnan().any():
            continue
        if key not in stats_dict:
            stats_dict[key] = defaultdict(Stats)
        for j in range(len(t)):
            stats_dict[key][j].add_data(t[j].item())
    return stats_dict

def get_model_path(model_family='llama', max_size='7b', min_size='125m'):
    if "LLM_DIR" not in os.environ:
        model_dir = ""
    else:
        # if you save checkpoints in your own path,
        # you should change this ENV and the following model_sigs.
        model_dir = os.environ['LLM_DIR']
        
    if model_family == 'llama':
        model_sigs = [
            'huggyllama/llama-7b',
            'huggyllama/llama-13b',
            'huggyllama/llama-30b',
            'huggyllama/llama-65b',
        ]
    elif model_family == 'opt':
        model_sigs = ["opt-125m","opt-350m","opt-1.3b","opt-2.7b","opt-6.7b","opt-13b", "opt-30b",]
        model_sigs = [f"facebook/{ms}" for ms in model_sigs]
    else:
        raise NotImplementedError()
    
    # check size
    def get_size_from_sig(ms):
        size = ms.split('-')[1]
        return size
    
    def check_max_size(ms, max_size):
        assert max_size.endswith('b') or max_size.endswith('m')
        ms_size = get_size_from_sig(ms)
        if max_size.endswith('b') and ms_size.endswith('m'):
            return True
        elif max_size.endswith('m') and ms_size.endswith('b'):
            return False
        else:
            return float(max_size[:-1]) >= float(ms_size[:-1])
    
    def check_min_size(ms, min_size):
        assert min_size.endswith('b') or min_size.endswith('m')
        ms_size = get_size_from_sig(ms)
        if min_size.endswith('m') and ms_size.endswith('b'):
            return True
        elif min_size.endswith('b') and ms_size.endswith('m'):
            return False
        else:
            return float(min_size[:-1]) <= float(ms_size[:-1])

    model_sigs = [ms for ms in model_sigs if check_max_size(ms, max_size)]
    model_sigs = [ms for ms in model_sigs if check_min_size(ms, min_size)]

    return model_dir, model_sigs

def read_file(document_path):
    with open(document_path, 'r') as f:
        documents = f.readlines()
        
    # if directly read in index file, "idx" should be in the document name.
    if 'idx' in document_path:
        documents = [[int(item) for item in doc.strip().split()] for doc in documents]
    else:
        documents = [doc.rstrip() for doc in documents]
    return documents

@torch.no_grad()
def compute_stats_dict(single_eval_func, document_path, model, tokenizer, **kwargs):
    stats_dict = {}
    cnt = 0
    documents = read_file(document_path)
    for seleteced_sentence_id, rep_sentence in tqdm.tqdm(enumerate(documents)):
        cnt += 1
        # Random put a prefix and put repetitive sentences as pseudo repetitive genetations
        # You can delete it and can also observe the self-reinforcement effect
        prefix_sentence = sentence_list[seleteced_sentence_id % len(sentence_list)]['base_sentence']
        ret_dict = single_eval_func(
            prefix_sentence=prefix_sentence, 
            base_sentence=rep_sentence, 
            model=model, 
            tokenizer=tokenizer, 
            **kwargs
        )
        stats_dict = gather_and_update_all_metrics(stats_dict, ret_dict, tokenizer)

    return stats_dict

def get_data_file(data_dir, model_sigs):
    wiki_doc = os.path.join(data_dir, "wiki_sentences.txt.detok")
    if 'llama' in model_sigs[0]:
        random_doc = os.path.join(data_dir, "random.llama.idx")
    elif 'opt' in model_sigs[0]:
        random_doc = os.path.join(data_dir, "random.opt.idx")
    else:
        random_doc = os.path.join(data_dir, "random.gpt.idx")
    book_doc = os.path.join(data_dir, "book_large_p1.txt.sample_10k.detok.filter_1k")
    docs = [wiki_doc, random_doc, book_doc]
    keys = ['wiki', 'random', 'book']
    return docs, keys

def maybe_encode_sentence(tokenizer, sentence, device='cuda', add_special_tokens=True):
    if isinstance(sentence, list):
        # skip tokenize and convert to tensor if already tokenized.
        encoded_sentence = torch.tensor(sentence)[None].long().to(device)
    elif isinstance(sentence, str):
        encoded_sentence = tokenizer.encode(sentence, truncation=False, return_tensors="pt", add_special_tokens=add_special_tokens).to(device)
    else:
        raise
    return encoded_sentence

def gather_neighbours_of_targets(targets, index_mask, similarities, neighbour_size=5):
    # Case A: targets [bsz, l]; index_mask [bsz, l]
    # Case B: targets [bsz, l, K]; index_mask [bsz, l]
    assert targets.ndim in [2, 3]
    assert index_mask.ndim == 2

    target_sims = similarities[targets[index_mask>0]]
    similar_tokens = torch.topk(target_sims, k=neighbour_size, dim=-1)[1] # [N_target, (K), N_neighbour]
    # expand another dimension for storing neighbour tokens
    # target_shape = list(targets.shape)
    new_target_shape = list(targets.shape) + [neighbour_size, ]
    neighbour_target_tokens = targets.unsqueeze(-1).expand(new_target_shape).clone() # [1, L, (K), N_n]
    # set to similar tokens
    neighbour_target_tokens[index_mask>0] = similar_tokens.to(neighbour_target_tokens.device)
    # get new target tokens
    new_targets = torch.cat([targets.unsqueeze(-1), neighbour_target_tokens], dim=-1) # [1, L, *, N_n+1]
    # reshape to ndim == 3
    if targets.ndim == 3:
        bsz, l = targets.shape[:2]
        targets = new_targets.reshape([bsz, l, -1])
    else:
        targets = new_targets
        
    return targets
        
def maybe_get_similarity_matrix(model, similarity_measure='dot'):
    # pre-compute the similarity matrix.
    if similarity_measure == 'dot':
        similarity_matrix = torch.matmul(model.lm_head.weight, model.lm_head.weight.t()) # [V, V]
    elif similarity_measure == 'cos':
        similarity_matrix = torch.matmul(model.lm_head.weight, model.lm_head.weight.t()) # [V, V]
        l2_norm = torch.norm(model.lm_head.weight, dim=1) # [V]
        similarity_matrix = similarity_matrix / (l2_norm[None] * l2_norm[:, None])
    else:
        raise NotImplementedError()
    # NOTE: set diagonal entries to -inf
    V = similarity_matrix.shape[-1]
    similarity_matrix[torch.arange(V),torch.arange(V)] = float('-inf')
    return similarity_matrix
