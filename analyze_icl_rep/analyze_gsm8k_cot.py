import torch
from analyze_icl_rep.utils import *
import tqdm
import pickle
import gc
import pandas as pd
import numpy as np
from collections import defaultdict

def set_seeds(seed):
    np.random.seed(seed)

    # Set the seed for pandas
    # pd.set_option('mode.chained_assignment', None)
    # pd.np.random.seed(seed)

    # Set the seed for PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available(): # If you're using a GPU
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return

# analyze MMLU dataset
def build_custom_parser(parser):
    parser.add_argument('--include_val', action='store_true', help="whether to include validation set as a part of demonstrations.")
    parser.add_argument('--ntrain', type=int, default=5)
    parser.add_argument('--ignore_task_prompt', action='store_true', help="whether to exclude task prompt when testing.")
    parser.add_argument('--delimiter', type=str, default='standard')
    parser.add_argument('--data_dir', type=str, default='ICL/gsm8k//')
    parser.add_argument('--cache_dir', type=str, default='')
    parser.add_argument('--hf_cache_dir', type=str, default='')
    parser.add_argument('--test_normal', action='store_true', help="whether to include validation set as a part of demonstrations.")
    parser.add_argument('--mask_answer', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--demo_type', type=str, default='hub', choices=['original','hub'])

    return parser

# def format_subject(subject):
#     l = subject.split("_")
#     s = ""
#     for entry in l:
#         s += " " + entry
#     return s.strip()

# def gen_delimiter(args):
#     if args.delimiter == 'standard':
#         delimiter = "\nAnswer:"
#     elif args.delimiter.startswith('repeat'):
#         # repeat_n
#         rep_times = int(args.delimiter.split('_')[-1])
#         delimiter = "\nAnswer:"*rep_times
#     elif args.delimiter.startswith('rand_char'):
#         # random char from 26 english chars
#         import string, random
#         n = int(args.delimiter.split('_')[-1])
#         delimiter = random.choices(string.ascii_letters, k=n)
#         delimiter = "\n"+"".join(delimiter)
#     elif args.delimiter.startswith('rand_int'):
#         import random
#         n = int(args.delimiter.split('_')[-1])
#         delimiter = random.choices(list(map(str, range(0,10))), k=n)
#         delimiter = "\n"+"".join(delimiter)
#     else: raise NotImplementedError()
#     return delimiter

def replace_text_with_random_tokens(tokenizer, prompt):
    import random
    encoded = tokenizer.encode(
        prompt,
        add_special_tokens=False,
        truncation=False,
        return_tensors="pt",
    )
    # Get a list of all the unique tokens in the tokenizer
    all_tokens = list(set(tokenizer.get_vocab().keys()))
    
    # Remove the special tokens from the list of possible replacement tokens
    special_tokens = tokenizer.all_special_tokens
    possible_tokens = [t for t in all_tokens if t not in special_tokens]
    replacement_token = random.choices(possible_tokens, k=encoded.shape[1])
    random_encoded = tokenizer.convert_tokens_to_ids(replacement_token)
    replaced_sentence = tokenizer.decode(random_encoded)
    return replaced_sentence

def replace_idx_with_random_tokens(tokenizer, tensor, args):
    # here, hard code to avoid special tokens is time-efficient...
    if args.model_family == 'opt':
        start_id = 4
    elif args.model_family == 'llama':
        start_id = 3
    else:
        raise NotImplementedError()
    random_encoded = torch.randint_like(tensor, low=start_id, high=len(tokenizer))
    assert random_encoded.shape[0] == 1
    replaced_sentence = tokenizer.decode(random_encoded[0])
    return replaced_sentence

def format_example_masked_with_spec(tokenizer, demon_t, demon_spec, args):
    """Generate an masked demonstration with given tokenized demo and demo_spec. Customized for GSM8k, different from the one used in MMLU dataset.

    Args:
        tokenizer (Tokenizer): Transformers tokenizer.
        demon_t (List): List of demonstrations options.
        demon_spec (Dict): Specification about whether the entry is tokenized. 
        args: all options

    Returns:
        _type_: _description_
    """
    delimiters = get_delimiters(args)
    prompt = ""
    assert len(demon_t)+1 == len(delimiters)
    for i in range(len(demon_t)):
        prompt += delimiters[i]
        # the last choice should add an extra space.
        cur = demon_t[i]
        if demon_spec[i] is True:
            prompt += replace_idx_with_random_tokens(tokenizer, cur, args)
        else:
            prompt += cur
    prompt += delimiters[len(demon_t)]
    return prompt

def get_delimiters(args):
    if args.demo_type == 'hub':
        ret = ["Question: ", "\n", "\n\n"]
    else:
        # manually add the pattern
        ret = ["Question: ", "\nLet's think step by step\n", "\n\n"]
    if args.delimiter == 'no_q_pattern':
        ret[0] = ""
    elif args.delimiter == 'no_qa_split':
        ret[1] = ""
    elif args.delimiter == 'no_demos_split':
        ret[2] = ""
    elif args.delimiter == 'no_qp_split':
        ret[1] = ret[1][1:] # remove \n
    return ret

def get_tokenized_demonstrations(tokenizer, args, dev_df):
    ps = []
    k, n = dev_df.shape[0], dev_df.shape[1]
    if args.test_normal is True:
        # set mask spec to all False when testing normal cases.
        convert_spec = {key: False for key in range(0, 2)}
    else:
        convert_spec = {
            0: True,
            1: False, 
        }
    if args.mask_answer is True: convert_spec[1] = True
    
    for i in tqdm.tqdm(range(k)):
        tmp = []
        # NOTE: each demonstrations has several fields
        # question, options should be tokenized, answer should not.
        # (Q, Answer)
        try:
            for j in range(n):
                if convert_spec[j] is True:
                    tok = tokenizer.encode(
                        dev_df.iloc[i].iloc[j],
                        add_special_tokens=False,
                        truncation=False,
                        return_tensors="pt",
                    )
                else:
                    tok = dev_df.iloc[i].iloc[j]
                tmp.append(tok)
        except:
            # in rare cases, dev_df has weird nan. skip these examples. They will not affect following examples
            continue
        ps.append(tmp)
    return ps, convert_spec

def format_example(df, include_answer=True):
    # delimiters = get_delimiters(args)
    prompt = f"Question: {df.iloc[0]}"
    prompt += "\n"
    if include_answer:
        # should not include answer in this script
        # raise
        prompt += "{}\n\n".format(df.iloc[1])
    return prompt

def gen_masked_prompt_from_tok_demons(tokenizer, test_df, demons_t, demons_t_spec, args):
    ps = []
    # NOTE: no task prompt for GSM8k
    k = len(demons_t)
    for i in range(k): 
        ps.append(format_example_masked_with_spec(tokenizer, demon_t=demons_t[i], demon_spec=demons_t_spec, args=args))
    # ps.append(format_example(test_df, include_answer=True))
    test_spec = {key: False for key in range(0, 2)}
    ps.append(format_example_masked_with_spec(tokenizer, test_df, demon_spec=test_spec, args=args))
    return ps

@torch.no_grad()
def single_icl_sample_cot_pattern(model, tokenizer, prompt_df, args):
    device = 'cuda'
    assert prompt_df.shape[0] >= args.ntrain + 1
    prompt_lst = [prompt_df.iloc[i] for i in range(args.ntrain)] + [prompt_df.iloc[-1]]
    # try:
    #     prompt = "".join(prompt_lst)
    #     inputs = maybe_encode_sentence(tokenizer, prompt)
    # except:
    #     print('Meet errors.')
    #     # weird nan also appear here. 
    #     return {}
    
    # concat CoT pattern
    pattern = "\nLet's think step by step\n"
    prompt = "".join(prompt_lst)
    # split prompt by pattern
    p_splits = prompt.split(pattern)
    tmp = []
    pattern_inputs = maybe_encode_sentence(tokenizer, pattern, add_special_tokens=False)
    p_idx_m, idx_m = [], []
    # construct index mask and pattern index mask
    for i, p in enumerate(p_splits[:-1]):
        if i != 0:
            tmp.append(pattern_inputs)
            p_idx_m.append(torch.zeros_like(pattern_inputs))
            idx_m.append(torch.zeros_like(pattern_inputs))
            
        p_t = maybe_encode_sentence(tokenizer, p, add_special_tokens=False)
        tmp.append(p_t)
        # except the last
        p_idx_m.append(torch.zeros_like(p_t))
        idx_m.append(torch.zeros_like(p_t))
        
    # deal with the last one
    tmp.append(pattern_inputs)
    p_idx_m.append(torch.ones_like(pattern_inputs))
    idx_m.append(torch.zeros_like(pattern_inputs))
    
    p_last = maybe_encode_sentence(tokenizer, p_splits[-1], add_special_tokens=False)
    tmp.append(p_last)
    idx_m.append(torch.ones_like(p_last))
    p_idx_m.append(torch.zeros_like(p_last))

    inputs = torch.cat(tmp, dim=1)
    pattern_index_mask = torch.cat(p_idx_m, dim=1)
    full_index_mask = torch.cat(idx_m, dim=1)

    # duplicate several times
    outputs = model(inputs)
    logits = outputs.logits # [bsz, L, V]
    probs = torch.softmax(logits, dim=-1) # [bsz, L, V]
    origin_probs = probs.clone()
    
    # get full index mask and full target tokens
    # full_index_mask = torch.zeros_like(inputs)
    # full_index_mask[:, -1] = 1
    full_index_mask = shift_left(full_index_mask, pad=False)
    pattern_index_mask = shift_left(pattern_index_mask, pad=False)
    full_target_tokens = shift_left(inputs, pad=tokenizer.pad_token_id)

    results_dict = {
        "base_tokens": None,
        "target_tokens": None,
        "input_tokens": inputs.cpu(),
        "predicted_max_probs": None,
        "split_logits": None,
        "split_probs": None,
        "full_probs": origin_probs.cpu(),
        "full_targets": full_target_tokens.cpu(),
        "full_index_mask": full_index_mask.cpu(),
        "pattern_index_mask": pattern_index_mask.cpu(),
    }
    
    return results_dict

def encode_strings(tokenizer, strings):
    """
    Encode a list of strings into token indexes using a given tokenizer.

    Args:
        tokenizer: The tokenizer to use.
        strings: A list of strings to encode.

    Returns:
        A list of encoded strings, where each string is a list of token indexes.
    """

    encoded_strings = []

    # Loop over each string in the input list
    for string in strings:
        # The tokenizer.encode function converts a text string into a sequence of token indexes.
        # It first tokenizes the text into subwords, then converts each subword into its corresponding token index.
        # NOTE: should add prefix space or not? How about sentencepiece tokenizers? We should add G'?
        encoded_string = [tokenizer.encode(string, add_special_tokens=False)]

        # Add the encoded string to the list
        encoded_strings.extend(encoded_string)

    return encoded_strings

@torch.no_grad()
def compute_stats_dict_gsm8k(task_name, model, tokenizer, prompts_df, **kwargs):
    stats_dict = {}
    # cnt = 0
    print('Testing %s ...' % task_name)

    for i in tqdm.tqdm(range(prompts_df.shape[0])):
        ret_dict = single_icl_sample_cot_pattern(
            model=model, 
            tokenizer=tokenizer,
            prompt_df=prompts_df.iloc[i],
            **kwargs
        )
        if ret_dict: # if get an empty return, we should skip updates
            stats_dict = gather_and_update_all_metrics(stats_dict, ret_dict, tokenizer)

    return stats_dict

def pre_tokenize_mask_data(tokenizer, args):
    prompt_dict = defaultdict(list)
    # for task_name in tasks:
    task_name = "gsm8k"
    print('Processing Task: {}'.format(task_name))
    args.cur_task = task_name
    test_df = pd.read_json(f"{args.data_dir}/data/test.jsonl", lines=True)

    # read dataset
    if args.demo_type == 'hub':
        # in text format
        prompt_original = open(f'{args.data_dir}/lib_prompt/prompt_original.txt').read()
        # split demonstrations
        demons = prompt_original.split('\n\n')
        demons = [item.replace("Question: ", "").split("\n", 1) for item in demons]
        # create demons dataframe
        dev_df = pd.DataFrame(demons)

    elif args.demo_type == 'original':
        train_df = pd.read_json(f"{args.data_dir}/data/train.jsonl", lines=True)
        # split part to be dev_df
        from sklearn.model_selection import train_test_split
        _, dev_df = train_test_split(train_df, test_size=50, random_state=args.seed)
        # read demos from train and test.
    else:
        raise

    # first encode all demonstrations
    print('Tokenize all demonstrations!')
    demons_t, demons_t_spec = get_tokenized_demonstrations(tokenizer, args, dev_df)

    print('Generate and cache masked prompts!')
    prompts = []
    for i in tqdm.tqdm(range(test_df.shape[0])):
        ps = gen_masked_prompt_from_tok_demons(tokenizer, test_df.iloc[i], demons_t, demons_t_spec, args)
        prompts.append(ps)

    prompts_df = pd.DataFrame(prompts)
    prompt_dict[task_name] = prompts_df
    return prompt_dict

def main(args):
    set_seeds(args.seed)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    model_dir, model_sigs = get_model_path(args.model_family, args.model_max_size, args.model_min_size)
    print(model_sigs)
    
    output_dir = args.output_dir
    
    def process(mp):
        task = 'gsm8k'
        args.cur_task = task
        f_path = os.path.join(output_dir, "{}_stats_{}_{}_{}_{}_{}-cot-ndem-igtask-del-seed.test.pkl".format(mp, task, args.ntrain, args.ignore_task_prompt, args.delimiter, args.seed))
        if os.path.exists(f_path):
            print("Skip {}".format(f_path))
            return

        model_path = os.path.join(model_dir, mp)
        model, tokenizer = load_model_and_tokenizer(model_path)
        
        prompt_dict = pre_tokenize_mask_data(tokenizer, args)
        
        stat_dict = compute_stats_dict_gsm8k(task, model, tokenizer, prompts_df=prompt_dict[task], args=args)
        with open(f_path, 'wb') as f:
            pickle.dump(stat_dict, f)
            
        del model
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        return

    for mp in model_sigs[:]:
        process(mp)
        gc.collect()

if __name__ == '__main__':
    parser = build_common_parser()
    parser = build_custom_parser(parser)
    args = parser.parse_args()

    main(args)