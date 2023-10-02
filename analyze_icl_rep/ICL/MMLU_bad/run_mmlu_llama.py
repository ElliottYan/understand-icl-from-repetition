import json
import os
import time 
from tqdm import tqdm
import argparse
import pandas as pd
import random
# random.seed(0)
import torch
import transformers
from datasets import load_dataset
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModel, AutoModelForCausalLM
import tensor_parallel as tp
import accelerate
from pathlib import Path
from typing import Tuple
import numpy as np


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

def compute_metric(output_filename):
    with open(output_filename, 'r') as f:
        run_results = json.load(f)
    with open(output_filename.replace(".json", ".txt"), 'w', encoding="utf8") as f:
        total_acc = 0
        total_fake_acc = 0
        total_num = 0
        aaa = 0
        bbb = 0
        ccc = 0
        ddd = 0
        others = 0
        for task in run_results:
            acc = 0
            fake_acc = 0
            pred_answers = run_results[task]['pred_answers']
            gold_answers = run_results[task]['gold_answers']
            fake_answers = run_results[task]['fake_answers']
            for pred, gold, fake in zip(pred_answers, gold_answers, fake_answers):
                if pred == gold: acc += 1
                elif pred != gold and pred == fake:
                    fake_acc += 1
                if pred=="A":
                    aaa += 1
                elif pred=="B":
                    bbb += 1
                elif pred=="C":
                    ccc += 1
                elif pred=="D":
                    ddd += 1
                else:
                    others += 1
            # print("ACC-%s: %.4f" % (task, acc/len(gold_answers)))
            # f.write("ACC-%s: %.4f" % (task, acc/len(gold_answers)) +"\n")
            total_acc += acc
            # print("Fake_ACC-%s: %.4f" % (task, fake_acc / len(fake_answers)))
            # f.write("Fake_ACC-%s: %.4f" % (task, fake_acc / len(fake_answers)) +"\n")
            total_fake_acc += fake_acc
            total_num += len(gold_answers)
        total_num = float(total_num)
        print("\n\n")
        f.write("\n\n" +"\n")
        print("Total_num: %.4f" % (total_num))
        f.write("Total_num: %.4f" % (total_num) +"\n")
        print("ACC-all: %.4f" % (total_acc/total_num))
        f.write("ACC-all: %.4f" % (total_acc/total_num) +"\n")
        print("Fake_ACC-all: %.4f" % (total_fake_acc / total_num))
        f.write("Fake_ACC-all: %.4f" % (total_fake_acc / total_num) +"\n")
        print("A-prop: %.4f" % (aaa / total_num))
        f.write("A-prop: %.4f" % (aaa / total_num) +"\n")
        print("B-prop: %.4f" % (bbb / total_num))
        f.write("B-prop: %.4f" % (bbb / total_num) +"\n")
        print("C-prop: %.4f" % (ccc / total_num))
        f.write("C-prop: %.4f" % (ccc / total_num) +"\n")
        print("D-prop: %.4f" % (ddd / total_num))
        f.write("D-prop: %.4f" % (ddd / total_num) +"\n")
        print("Others-prop: %.4f" % (others / total_num))
        f.write("Others-prop: %.4f" % (others / total_num) +"\n")

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

def format_example(df, idx, exp_type, exp_subtype, include_answer=True, nonsense=""):
    # nonsense_map = {
    #     "A": "\nPlease answer the question.",
    #     "B": "\nKindly provide your response.",
    #     "C": "\nCurious to know your thought.",
    #     "D": "\nLooking forward to your choice.",
    # }
    # nonsense_pool = {
    #     "\nPlease answer the question.",
    #     "\nKindly provide your response.",
    #     "\nCurious to know your thought.",
    #     "\nLooking forward to your choice.",
    # }
    
    reorder_idx_map = {
        "A": 1,
        "B": 2,
        "C": 3,
        "D": 4,
    }
    fake_answer = None
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    if (exp_type == "reorder" or exp_type == "repeat") and include_answer:
        fake_answer = exp_subtype
        gold_col = reorder_idx_map[str(df.iloc[idx, k + 1])]
        fake_col = reorder_idx_map[fake_answer]
        gold_val = df.at[idx, gold_col]
        df.at[idx, gold_col] = df.at[idx, fake_col]
        df.at[idx, fake_col] = gold_val
        df.at[idx, k + 1] = fake_answer
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j+1])
    if exp_type.startswith("nonsense"):
        if include_answer:
            prompt += nonsense #nonsense_map[str(df.iloc[idx, k + 1])]
        else:
            fake_answer = exp_subtype
            prompt += nonsense # nonsense_map[fake_answer]
    if exp_type == "7X":
        prompt += "\n" + 7*exp_subtype
        fake_answer = exp_subtype
    else:
        prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
        if fake_answer is None: # for normal-1
            fake_answer = str(df.iloc[idx, k + 1])
    return prompt, fake_answer

def get_nonsense():
    nonsense_pool = [
        "\nPlease answer the question.",
        "\nKindly provide your response.",
        "\nCurious to know your thought.",
        "\nLooking forward to your choice.",
    ]
    nonsense = random.sample(nonsense_pool, 1)[0]
    return nonsense

# def gen_prompt(train_df, subject, exp_type, exp_subtype, k=-1):
#     prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))
#     if k == -1:
#         k = train_df.shape[0]
#     for i in range(k):
#         if exp_type == "repeat":
#             prompt_cont, fake_answer = format_example(train_df, 0, exp_type, exp_subtype)
#             prompt += prompt_cont
#         else:
#             prompt_cont, fake_answer = format_example(train_df, i, exp_type, exp_subtype)
#             prompt += prompt_cont
#     return prompt, fake_answer

def prepare_input(tokenizer, prompts):
    input_tokens = tokenizer.batch_encode_plus(prompts, return_tensors="pt", padding=True)
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to('cuda')

    return input_tokens

def load(ckpt_dir):
    n_gpus = torch.cuda.device_count()
    
    if 'llama' in ckpt_dir:
        # we use tensor parallel for loading llama
        # model = LlamaForCausalLM.from_pretrained(ckpt_dir, low_cpu_mem_usage = True, torch_dtype=torch.float16)
        # model = tp.tensor_parallel(model, [i for i in range(n_gpus)]) 
        model = AutoModelForCausalLM.from_pretrained(ckpt_dir, device_map = 'auto', torch_dtype=torch.float16, trust_remote_code=True)
        model.tie_weights()
        tokenizer = LlamaTokenizer.from_pretrained(
        ckpt_dir,
        use_fast=False,
        padding_side="left",
        )
        # tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
        # tokenizer.bos_token_id = 1
    else:
        model = AutoModelForCausalLM.from_pretrained(ckpt_dir, device_map = 'auto', torch_dtype=torch.float16, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, padding_side='left')
        
    model.eval()
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def batch_split(prompts, batch_num):
    batch_prompts = []
    mini_batch = []
    for prompt in prompts:
        mini_batch.append(prompt)
        if len(mini_batch) == batch_num:
            batch_prompts.append(mini_batch)
            mini_batch = []
    if len(mini_batch) != 0:
        batch_prompts.append(mini_batch)
    return batch_prompts

def batch_infer(model, tokenizer, prompts, batch_size):
    answers = []
    answers_scores = []
    for batch_input in tqdm(batch_split(prompts, batch_size)):
        encode_inputs = prepare_input(tokenizer, batch_input)
        outputs = model.generate(**encode_inputs, max_new_tokens=1, output_scores=True, return_dict_in_generate=True)
        # import pdb; pdb.set_trace()
        scores = outputs.scores[0]#.tolist()
        answers_scores.append(scores)
        answers.extend(tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True))
    answers = [answer[-1] for answer in answers]
    answers_scores = torch.cat(answers_scores, dim=0).cpu()
    return answers, answers_scores

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    # Set the seed for pandas
    # pd.np.random.seed(seed)

    # Set the seed for PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using CUDA

    # Additional steps for reproducibility in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(model, tokenizer, args, exp_type, exp_subtype):
    set_seed(args.seed)
    
    print('Start exp -- exp_type: {}, exp_subtype: {}'.format(exp_type, exp_subtype))
    
    run_results = {}
    ckpt_base = os.path.basename(args.ckpt_dir)
        
    output_filename = '%s/run_results_%s_%s_%s_demo%s_seed%s.json' % (args.output_dir, ckpt_base, exp_type, exp_subtype, args.ntrain, args.seed)
    print(output_filename)
    if os.path.exists(output_filename):
        print('Already existed {} .'.format(output_filename))
        return
    
    start_time = time.time()
    for task in TASKS:
        print('Testing %s ...' % task)
        if args.test_dir != "": test_dir = args.test_dir
        else: test_dir = os.path.join(args.data_dir, "test")
        records = []
        # if args.exp_allA_balanced: dev_suffix
        dev_suffix = args.dev_suffix
        test_suffix = args.test_suffix
        if args.exp_allA_balanced:
            dev_suffix = '_allA.csv'
            test_suffix = '.balanced.csv'
        
        dev_path = os.path.join(args.data_dir, "dev", f"{task}_dev{dev_suffix}")
        val_path = os.path.join(args.data_dir, "val", f"{task}_val{dev_suffix}")
        test_path = os.path.join(test_dir, f"{task}_test{test_suffix}")
        print(dev_path)
        print(val_path)
        print(test_path)

        dev_df = pd.read_csv(dev_path, header=None)
        if args.include_val:
            val_df = pd.read_csv(val_path, header=None)
            dev_df = pd.concat([dev_df, val_df], ignore_index=True)

        test_df = pd.read_csv(test_path, header=None)

        #test_df = pd.read_csv(os.path.join(args.data_dir, "test", task + "_test.csv"), header=None)
        # random index
        pdx_cache_path = os.path.join(args.cache_dir, "{}.seed{}.pidx.npy".format(task, args.seed))
        if os.path.exists(pdx_cache_path):
            print(f'Read pdx from cache {pdx_cache_path}.')
            pdx = np.load(pdx_cache_path)
        else:
            # control demonstrations for each seed.
            pdx = np.stack([np.random.choice(dev_df.shape[0], dev_df.shape[0], replace=False) for _ in range(test_df.shape[0])])
            np.save(pdx_cache_path, pdx)

        for i in range(test_df.shape[0]):
        # for i in range(5): ##############
            # get prompt and make sure it fits
            k = args.ntrain
            # select demonstrations with pdx
            demo_df = dev_df.iloc[pdx[i]]
            # if exp_type == "nonsense":
            #     train_prompt, _ = gen_prompt(demo_df, task, exp_type, exp_subtype, k) 
            # else:
            #     train_prompt, fake_answer = gen_prompt(demo_df, task, exp_type, exp_subtype, k)
            prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(task))
            # select a nonsense phrase
            
            nonsense = get_nonsense() if exp_type.startswith('nonsense') else ""
                
            for j in range(k):
                if demo_df.iloc[j, 5] == exp_subtype:
                    cur_nonsense = nonsense if args.exp_type != "nonsense_test" else ""
                else:
                    cur_nonsense = ""
                if exp_type == "repeat":
                    prompt_cont, fake_answer = format_example(demo_df, 0, exp_type, exp_subtype)
                    prompt += prompt_cont
                else:
                    prompt_cont, fake_answer = format_example(demo_df, j, exp_type, exp_subtype, nonsense=cur_nonsense)
                    prompt += prompt_cont
            # NOTE: for nonsense, fake_answer is set when formatting prompt_end??
            
            prompt_end, fake_answer = format_example(test_df, i, exp_type, exp_subtype, include_answer=False, nonsense=nonsense)
            
            prompt = prompt + prompt_end
            while len(tokenizer.tokenize(prompt)) + 1> 2048: # bos token
                prompt_split = prompt.split("\n\n")
                prompt_split.pop(1)
                prompt = '\n\n'.join(prompt_split)
            label = test_df.iloc[i, test_df.shape[1]-1]
            # 这种情况下fake acc应该为0
            if exp_type == "normal" and args.ntrain == 5:
                fake_answer = label
            record = {'prompt':prompt, 'answer':label, 'fake_answer':fake_answer}
            records.append(record)

        prompts = [record['prompt'] for record in records]
        pred_answers, pred_answers_scores = batch_infer(model, tokenizer, prompts, args.batch_size)
        gold_answers = [record['answer'] for record in records]
        fake_answers = [record['fake_answer'] for record in records]
        run_results[task] = {'prompts':prompts, 'pred_answers':pred_answers, 'gold_answers':gold_answers, "fake_answers":fake_answers}
        # save pts
        torch.save(pred_answers_scores, "{}.{}.pt".format(output_filename, task))
        # run_results[task] = {'prompts':prompts, 'pred_answers':pred_answers, 'pred_answers_scores':pred_answers_scores, 'gold_answers':gold_answers, "fake_answers":fake_answers}
    with open(output_filename, 'w') as f:
        json.dump(run_results, f, ensure_ascii=False, indent=2)
    
    compute_metric(output_filename)
    end_time = time.time()
    print("total run time %.2f" % (end_time - start_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, default='models/llama-7b')
    # parser.add_argument('--param_size', type=str, default='7')
    # parser.add_argument('--model_type', type=str, default='llama')
    parser.add_argument('--exp_type', type=str, default='7X', choices=["normal", "7X", "nonsense", "nonsense-only-trigger", "repeat", "reorder", "all", "normal_and_nonsense", "nonsense_test"])
    parser.add_argument('--exp_subtype', type=str, default="None", choices=["A", "B", "C", "D", "None"])
    parser.add_argument('--ntrain', type=int, default=5)
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--test_dir', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--output_dir', type=str, default='predictions/')
    parser.add_argument('--include_val', action='store_true', help="whether to include validation set as a part of demonstrations.")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cache_dir', type=str, default='./prompt_cache/')
    parser.add_argument('--exp_allA_balanced', action='store_true')
    parser.add_argument('--dev_suffix', type=str, default=".csv")
    parser.add_argument('--test_suffix', type=str, default=".csv")
    
    
    args = parser.parse_args()
    if args.exp_subtype == "None":
        args.exp_subtype = None

    model, tokenizer = load(args.ckpt_dir)
    
    if args.exp_type == 'all':
        for exp_type in ["normal", "7X", "nonsense", "repeat", "reorder"]:
            if exp_type != "normal":
                for exp_subtype in ["A", "B", "C", "D"]:
                    
                    main(model, tokenizer, args, exp_type, exp_subtype)
            else:
                main(model, tokenizer, args, exp_type, None)
    elif args.exp_type == 'normal_and_nonsense':
        for exp_type in ["nonsense", "normal"]:
            if exp_type != "normal":
                if args.exp_subtype == 'None':
                    for exp_subtype in ["A", "B", "C", "D"]:
                        main(model, tokenizer, args, exp_type, exp_subtype)
                else:
                    main(model, tokenizer, args, exp_type, args.exp_subtype)
            else:
                main(model, tokenizer, args, exp_type, None)
    else:
        main(model, tokenizer, args, args.exp_type, args.exp_subtype)
