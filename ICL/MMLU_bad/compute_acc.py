import torch
import os
import pandas as pd
import sys
import transformers
from transformers import LlamaTokenizer
import torch.nn.functional as F
import tqdm

cur, tok_path = sys.argv[1], sys.argv[2]

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

choices = ["▁A", "▁B", "▁C", "▁D"]
names = ["A", "B", "C", "D"]
i2c = {i:names[i] for i in range(len(names))}
c2i = {names[i]:i for i in range(len(names))}


tokenizer = LlamaTokenizer.from_pretrained(
    tok_path,
    use_fast=False,
    padding_side="left",
)

choice_ids = []
for choice in choices:
    ids = tokenizer.convert_tokens_to_ids(choice)
    choice_ids.append(ids)

corr = [0] * len(choices)
tott = [0] * len(choices)

cur_ot = []
baseline_ot = []
choices_t = torch.tensor(choice_ids)
cur_choice_ps = None
base_choice_ps = None

for task in TASKS:
# for task in tqdm.tqdm(TASKS):
    cur_fn = cur + f'.{task}.pt'
    # base_fn = baseline + f'.{task}.pt'
    cur_t = torch.load(cur_fn).cpu().float()
    # base_t = torch.load(base_fn)
    # assert cur_t.shape == base_t.shape
    cur_p = torch.softmax(cur_t, dim=-1)
    # base_p = torch.softmax(base_t, dim=-1)
    bsz = cur_p.shape[0]
    cur_choice_p = cur_p.gather(1, choices_t[None].expand(bsz, 4))
    # base_choice_p = base_p.gather(1, choices_t[None].expand(bsz, 4))
    if cur_choice_ps is None:
        cur_choice_ps = cur_choice_p
        # base_choice_ps = base_choice_p
    else:
        cur_choice_ps = torch.cat([cur_choice_ps, cur_choice_p], dim=0)
        # base_choice_ps = torch.cat([base_choice_ps, base_choice_p], dim=0)

# renormalize
cur_choice_ps = cur_choice_ps / cur_choice_ps.sum(-1, keepdims=True)
# base_choice_ps = base_choice_ps / base_choice_ps.sum(-1, keepdims=True)

test_dir = "self-reinforce-effect-icl/ICL/MMLU/data/test_sample_20"
# read dataset

all_tests = []
# for task in tqdm.tqdm(TASKS):
for task in TASKS:
    test_path = os.path.join(test_dir, task + "_test.balanced.csv")
    test_df = pd.read_csv(test_path, header=None)
    all_tests.append(test_df)

answers = pd.concat(all_tests, axis=0).iloc[:,-1]
        
tots = [0,] * 4
cors = [0,] * 4
assert answers.shape[0] == cur_choice_ps.shape[0]
# for j in range(len(choices)):
for i in range(answers.shape[0]):
    cur_gold = answers.iloc[i]
    cur_choice = cur_choice_ps[i].argmax().item()
    cur_ans = i2c[cur_choice]
    if cur_ans == cur_gold:
        cors[c2i[cur_ans]] += 1
    tots[c2i[cur_gold]] += 1

print([cors[k] / tots[k] for k in range(4)])

# base_ot = (1-base_choice_ps.sum(-1)).mean()
# cur_ot = (1-cur_choice_ps.sum(-1)).mean()
# print(f'off-target probs baseline: {base_ot}')
# print(f'off-target probs cur: {cur_ot}')
