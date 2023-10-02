import torch
import os
import pandas as pd
import sys
import transformers
from transformers import LlamaTokenizer
import torch.nn.functional as F
import tqdm

from sklearn.metrics import accuracy_score
from scipy.stats import ttest_rel, wilcoxon
from sklearn.model_selection import cross_val_score, KFold
import numpy as np

cur, base, tok_path = sys.argv[1], sys.argv[2], sys.argv[3]

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

def gather_all_predictions(prefix):
    cur_choice_ps = None
    for task in tqdm.tqdm(TASKS):
        cur_fn = prefix + f'.{task}.pt'
        cur_t = torch.load(cur_fn).cpu().float()
        # assert cur_t.shape == base_t.shape
        cur_p = torch.softmax(cur_t, dim=-1)
        bsz = cur_p.shape[0]
        cur_choice_p = cur_p.gather(1, choices_t[None].expand(bsz, 4))
        
        if cur_choice_ps is None:
            cur_choice_ps = cur_choice_p
        else:
            cur_choice_ps = torch.cat([cur_choice_ps, cur_choice_p], dim=0)
    cur_Y = cur_choice_ps.argmax(-1).numpy()
    return cur_Y

test_dir = "self-reinforce-effect-icl/ICL/MMLU/data/test_sample_20"
all_tests = []
# for task in tqdm.tqdm(TASKS):
for task in TASKS:
    test_path = os.path.join(test_dir, task + "_test.balanced.csv")
    test_df = pd.read_csv(test_path, header=None)
    all_tests.append(test_df)

answers = pd.concat(all_tests, axis=0).iloc[:,-1]
# gather answer indexs
gold_Y = []
for j in range(answers.shape[0]):
    y = c2i[answers.iloc[j]]
    gold_Y.append(y)

gold_Y = np.array(gold_Y)
# only choose D
# d_mask = gold_Y == 3
# choose ABC
d_mask = gold_Y != 3
gold_Y = gold_Y[d_mask]

cur_Y_lst = []
base_Y_lst = []
seeds = [0,1,2]
for seed in seeds:
    cur_s = cur.replace('[SEED]', f'{seed}')
    base_s = base.replace('[SEED]', f'{seed}')
    cur_s_Y = gather_all_predictions(cur_s)
    base_s_Y = gather_all_predictions(base_s)
    cur_s_Y = cur_s_Y[d_mask]
    base_s_Y = base_s_Y[d_mask]
    cur_Y_lst.append(cur_s_Y)
    base_Y_lst.append(base_s_Y)

    assert cur_s_Y.shape == gold_Y.shape
    assert base_s_Y.shape == gold_Y.shape

# Assuming clf1 is your baseline model, clf2 is your other model, and X, y are your data
cv = KFold(n_splits=5, shuffle=True, random_state=1)

# accs = []
cur_accs = []
base_accs = []
for i, (train_index, test_index) in enumerate(cv.split(gold_Y)):
    cur_acc, base_acc = 0, 0
    for j in range(len(seeds)):
        cur_acc += accuracy_score(gold_Y[test_index], cur_Y_lst[j][test_index])
        base_acc += accuracy_score(gold_Y[test_index], base_Y_lst[j][test_index])
    cur_accs.append(cur_acc / len(seeds))
    base_accs.append(base_acc / len(seeds))

print(cur_accs)
print(base_accs)
# Perform paired t-test
t_stat, p_val = ttest_rel(cur_accs, base_accs)

print(f'Paired t-test p-value: {p_val}')
print(t_stat)
# print(f'Wilcoxon test p-value: {w_p_val}')
