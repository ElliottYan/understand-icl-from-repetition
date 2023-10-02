import torch
# from utils import *
import utils
import os
import tqdm
import pickle
# import gc
# import argparse
# import json
import pandas as pd
import numpy as np
# import jsonlines
from collections import defaultdict

TASKS = [
        'abstract_algebra',
        'anatomy',
        'astronomy',
        'business_ethics',
        'clinical_knowledge',
        'college_biology',
        'college_chemistry',
        'college_computer_science',
        'college_mathematics',
        'college_medicine',
        'college_physics',
        'computer_security',
        'conceptual_physics',
        'econometrics',
        'electrical_engineering',
        'elementary_mathematics',
        'formal_logic',
        'global_facts',
        'high_school_biology',
        'high_school_chemistry',
        'high_school_computer_science',
        'high_school_european_history',
        'high_school_geography',
        'high_school_government_and_politics',
        'high_school_macroeconomics',
        'high_school_mathematics',
        'high_school_microeconomics',
        'high_school_physics',
        'high_school_psychology',
        'high_school_statistics',
        'high_school_us_history',
        'high_school_world_history',
        'human_aging',
        'human_sexuality',
        'international_law',
        'jurisprudence',
        'logical_fallacies',
        'machine_learning',
        'management',
        'marketing',
        'medical_genetics',
        'miscellaneous',
        'moral_disputes',
        'moral_scenarios',
        'nutrition',
        'philosophy',
        'prehistory',
        'professional_accounting',
        'professional_law',
        'professional_medicine',
        'professional_psychology',
        'public_relations',
        'security_studies', 
        'sociology',
        'us_foreign_policy',
        'virology',
        'world_religions']

choices = ["A", "B", "C", "D"]

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
    parser.add_argument('--task_name', type=str, default='all')
    parser.add_argument('--include_val', action='store_true', help="whether to include validation set as a part of demonstrations.")
    parser.add_argument('--ntrain', type=int, default=5)
    parser.add_argument('--ignore_task_prompt', action='store_true', help="whether to exclude task prompt when testing.")
    parser.add_argument('--delimiter', type=str, default='standard')
    parser.add_argument('--data_dir', type=str, default='ICL/MMLU/data/')
    parser.add_argument('--test_dir', type=str, default='')
    parser.add_argument('--cache_dir', type=str, default='./prompt_cache/')
    parser.add_argument('--test_normal', action='store_true', help="whether to include validation set as a part of demonstrations.")

    parser.add_argument('--seed', type=int, default=0)

    return parser

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s.strip()

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

# def format_example_masked(tokenizer, demon_t, spec, delimiter, include_answer=True):
#     prompt = replace_text_with_random_tokens(tokenizer, df.iloc[0])
#     k = df.shape[0] - 2
#     for j in range(k):
#         cur_choice = df.iloc[j+1]
#         cur_choice = replace_text_with_random_tokens(tokenizer, cur_choice)
            
#         prompt += "\n{}. {}".format(choices[j], cur_choice)
    
#     prompt += delimiter
#     prompt += " {}\n\n".format(df.iloc[k + 1])
#     return prompt

def get_option_idx(s):
    mapp = {
        'A':0, 'B':1, 'C':2, 'D':3
    }
    assert s in mapp
    return mapp[s]

def format_example_masked_with_spec(tokenizer, demon_t, demon_spec, args):
    """Generate an masked demonstration with given tokenized demo and demo_spec.

    Args:
        tokenizer (Tokenizer): Transformers tokenizer.
        demon_t (List): List of demonstrations options.
        demon_spec (Dict): Specification about whether the entry is tokenized. 
        args: all options

    Returns:
        _type_: _description_
    """
    delimiters, option_name, ans_indicator = get_delimiters(args)
    prompt = ""
    assert len(demon_t) == len(delimiters)
    for i in range(len(demon_t)):
        # the last choice should add an extra space.
        cur = demon_t[i]
        if i == len(demon_t)-1:
            # mapping the final choice
            assert isinstance(cur, str)
            opt_idx = get_option_idx(cur)
            cur = option_name[opt_idx][:-1]
            cur = " " + cur
        if demon_spec[i] is True:
            prompt += replace_idx_with_random_tokens(tokenizer, cur, args)
        else:
            prompt += cur
        prompt += delimiters[i]
    return prompt

def get_answer_indicator_pool():
    pool = [
        "\nSolution:",
        "\nReply:",
        "\nResponse:",
        "\nResult:",
        "\nChoice:"
    ]
    return pool

def get_answer_indicator_pool_v2():
    pool = [
        "\nSolution",
        "\nReply",
        "\nResponse",
        "\nResult",
        "\nChoice"
    ]
    return pool


def get_option_name_pool():
    pool = [
        ["1.", "2.", "3.", "4."],
        ["I.", "II.", "III.", "IV."],
        ["E.", "F.", "G.", "H."],
        ["(a).", "(b).", "(c).", "(d)."],
    ]
    return pool


def get_delimiters(args):
    answer_indicator_pool = get_answer_indicator_pool_v2()
    option_name_pool = get_option_name_pool()
    
    # random choice
    import random
    if args.delimiter == 'no_answer_indicator':
        ans_indicator = random.sample(answer_indicator_pool, 1)[0]
    else:
        ans_indicator = '\nAnswer:'
        
    if args.delimiter == 'no_option_name':
        option_names = random.sample(option_name_pool, 1)[0]
    else:
        option_names = ["A.", "B.", "C.", "D."]

    if args.delimiter == "remove_all":
        remove_all = True
    else:
        remove_all = False
    ret = []
    for i, choice in enumerate(choices):
        if args.delimiter == 'no_option_name' or remove_all is True:
            # ret.append("\n")
            ret.append("\n{}".format(option_names[i]))
        else:
            ret.append(f"\n{choice}. ")
    if args.delimiter == 'no_answer_indicator' or remove_all:
        # ret.append('\n')
        ret.append("{}".format(ans_indicator))
    else:
        ret.append("\nAnswer:")
    ret.append("\n\n")
    return ret, option_names, ans_indicator

# def gen_masked_prompt(tokenizer, test_df, dev_df, args):
#     ps = []
#     ps.append(get_task_prompt(args.cur_task))
#     delimiter = gen_delimiter(args)
#     # get icl prompt
#     k = dev_df.shape[0]
#     for i in range(k): 
#         try:
#             ps.append(format_example_masked(tokenizer, dev_df.iloc[i], delimiter))
#         except:
#             continue
#     # prompt += format_example_masked(tokenizer, test_df, i, delimiter, include_answer=False)
#     ps.append(format_example_masked(tokenizer, test_df, delimiter, include_answer=False, replace=False))
#     return ps

def get_tokenized_demonstrations(tokenizer, dev_df, args):
    ps = []
    k, n = dev_df.shape[0], dev_df.shape[1]
    if args.test_normal is True:
        # set mask spec to all False when testing normal cases.
        convert_spec = {key: False for key in range(0, 6)}
    else:
        convert_spec = {
            0: True,
            1: True, 
            2: True,
            3: True,
            4: True,
            5: False,
        }
    for i in tqdm.tqdm(range(k)):
        tmp = []
        # NOTE: each demonstrations has several fields
        # question, options should be tokenized, answer should not.
        # (Q, A, B, C, D, Answer)
        # try:
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
        # except:
        #     # in rare cases, dev_df has weird nan. skip these examples. They will not affect following examples
        #     continue
        ps.append(tmp)
    return ps, convert_spec

def get_task_prompt(task):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(task))
    return prompt

def format_example(df, include_answer=True):
    prompt = df.iloc[0]
    k = df.shape[0] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[j+1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[k + 1])
    return prompt
    
    # include args in the function call
    # delimiters = get_delimiters(args)
    # prompt = ""
    # assert len(df) == len(delimiters)
    # for i in range(len(df)):
    #     # the last choice should add an extra space.
    #     cur = df[i]
    #     if i == len(df)-1:
    #         assert isinstance(cur, str)
    #         cur = " " + cur
    #     if include_answer is False and i == len(df) - 1:
    #         break
    #     prompt += cur
    #     prompt += delimiters[i]
    # return prompt

def gen_masked_prompt_from_tok_demons(tokenizer, test_df, demons_t, demons_t_spec, args):
    ps = []
    ps.append(get_task_prompt(args.cur_task))
    k = len(demons_t)
    for i in range(k): 
        ps.append(format_example_masked_with_spec(tokenizer, demon_t=demons_t[i], demon_spec=demons_t_spec, args=args))
    ps.append(format_example(test_df, include_answer=False))
    return ps

@torch.no_grad()
def single_icl_sample_label_space_eval(model, tokenizer, prompt_df, choice_indexes, args):
    device = 'cuda'
    assert prompt_df.shape[0] >= args.ntrain + 1
    prompt_lst = [prompt_df.iloc[0], ] + [prompt_df.iloc[i+1] for i in range(args.ntrain)] + [prompt_df.iloc[-1]]
    if args.ignore_task_prompt:
        prompt_lst = prompt_lst[1:]
    try:
        prompt = "".join(prompt_lst)
        inputs = utils.maybe_encode_sentence(tokenizer, prompt)
    except:
        print('Meet errors.')
        # weird nan also appear here. 
        return {}
    
    # duplicate several times
    outputs = model(inputs)
    logits = outputs.logits # [bsz, L, V]
    probs = torch.softmax(logits, dim=-1) # [bsz, L, V]
    origin_probs = probs.clone()
    
    # get full index mask and full target tokens
    full_index_mask = torch.zeros_like(inputs)
    full_index_mask[:, -1] = 1
    full_target_tokens = torch.zeros_like(inputs).fill_(tokenizer.pad_token_id) # [1, L]
    full_target_tokens = full_target_tokens[:, :, None].expand(-1, -1, len(choice_indexes)).clone()
    choice_tensor = torch.tensor(choice_indexes).squeeze(-1).to(full_target_tokens.device)
    full_target_tokens[:, -1] = choice_tensor

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
def compute_stats_dict_mmlu(task_name, model, tokenizer, prompts_df, **kwargs):
    stats_dict = {}
    # cnt = 0
    print('Testing %s ...' % task_name)
    # get label ids
    label_ids = encode_strings(tokenizer, choices)

    for i in tqdm.tqdm(range(prompts_df.shape[0])):
        ret_dict = single_icl_sample_label_space_eval(
            model=model, 
            tokenizer=tokenizer,
            prompt_df=prompts_df.iloc[i],
            choice_indexes=label_ids,
            **kwargs
        )
        if ret_dict: # if get an empty return, we should skip updates
            stats_dict = utils.gather_and_update_all_metrics(stats_dict, ret_dict, tokenizer)

    return stats_dict

def pre_tokenize_mask_data(tasks, tokenizer, args):
    prompt_dict = defaultdict(list)
    for task_name in tasks:
        print('Processing Task: {}'.format(task_name))
        args.cur_task = task_name
        if args.test_normal: mask_sig = 'normal' 
        else: mask_sig = 'mask'
        cache_path = os.path.join(args.cache_dir, f"{mask_sig}-prompt_{task_name}_{args.model_family}_seed{args.seed}_del{args.delimiter}.csv")
        if os.path.exists(cache_path):
            print("Read cached file: {}".format(cache_path))
            prompts_df = pd.read_csv(cache_path, header=None, keep_default_na=False)
        else:        
            # read dataset
            dev_df = pd.read_csv(os.path.join(args.data_dir, "dev", task_name + "_dev.csv"), header=None, keep_default_na=False)
            if args.include_val:
                val_df = pd.read_csv(os.path.join(args.data_dir, "val", task_name + "_val.csv"), header=None, keep_default_na=False)
                dev_df = pd.concat([dev_df, val_df], ignore_index=True)
                
            # NOTE: add random sample demonstrations
            dev_df = dev_df.sample(n=dev_df.shape[0])
            # dev_df = 
            # dev_df = dev_df[:args.ntrain]
            if args.test_dir != "": test_dir = args.test_dir
            else: test_dir = os.path.join(args.data_dir, "test")
            test_df = pd.read_csv(os.path.join(test_dir, task_name + "_test.csv"), header=None, keep_default_na=False)

            # first encode all demonstrations
            print('Tokenize all demonstrations!')
            demons_t, demons_t_spec = get_tokenized_demonstrations(tokenizer, dev_df, args)
            
            print('Generate and cache masked prompts!')
            prompts = []
            for i in tqdm.tqdm(range(test_df.shape[0])):
                ps = gen_masked_prompt_from_tok_demons(tokenizer, test_df.iloc[i], demons_t, demons_t_spec, args)
                # generate normal prompt too.
                prompts.append(ps)
            prompts_df = pd.DataFrame(prompts)
            
            # cache generated prompts
            prompts_df.to_csv(cache_path, header=False, index=False)
        prompt_dict[task_name] = prompts_df
    return prompt_dict

def main(args):
    set_seeds(args.seed)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    model_dir, model_sigs = utils.get_model_path(args.model_family, args.model_max_size, args.model_min_size)
    print(model_sigs)
    
    output_dir = args.output_dir
    
    def process(mp):
        if args.task_name == 'all':
            all_tasks = TASKS
        else:
            assert args.task_name in TASKS
            all_tasks = [args.task_name, ]

        print(args)

        key = 'mmlu'
        # check all tasks are done before loading models.
        done = True
        for task in all_tasks:
            f_path = os.path.join(output_dir, "{}_stats_{}_{}_{}_{}_{}_{}-mc-task-ndem-igtask-del-seed.test.pkl".format(mp, key, task, args.ntrain, args.ignore_task_prompt, args.delimiter, args.seed))
            done = (done and os.path.exists(f_path))
        if done:
            print('All tasks are done. Exit.')
            return

        model_path = os.path.join(model_dir, mp)
        model, tokenizer = utils.load_model_and_tokenizer(model_path)

        prompt_dict = pre_tokenize_mask_data(all_tasks, tokenizer, args)

        for task in all_tasks:
            args.cur_task = task
            f_path = os.path.join(output_dir, "{}_stats_{}_{}_{}_{}_{}_{}-mc-task-ndem-igtask-del-seed.test.pkl".format(mp, key, task, args.ntrain, args.ignore_task_prompt, args.delimiter, args.seed))
            if os.path.exists(f_path):
                print("Skip {}".format(f_path))
                continue
            
            stat_dict = compute_stats_dict_mmlu(task, model, tokenizer, prompts_df=prompt_dict[task], args=args)
            with open(f_path, 'wb') as f:
                pickle.dump(stat_dict, f)
        del model
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        return

    for mp in model_sigs[:]:
        process(mp)
        # gc.collect()

if __name__ == '__main__':
    import time
    t1 = time.time()
    parser = utils.build_common_parser()
    parser = build_custom_parser(parser)
    args = parser.parse_args()
    t2 = time.time()
    print(f'Parse time {t2-t1}.')

    main(args)
    t3 = time.time()
    print(f'Job time {t3-t2}.')
