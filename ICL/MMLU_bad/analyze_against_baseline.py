import torch
import sys
import transformers
from transformers import LlamaTokenizer
import torch.nn.functional as F
import tqdm

cur, baseline, tok_path = sys.argv[1], sys.argv[2], sys.argv[3]

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

for task in tqdm.tqdm(TASKS):
    cur_fn = cur + f'.{task}.pt'
    base_fn = baseline + f'.{task}.pt'
    cur_t = torch.load(cur_fn)
    base_t = torch.load(base_fn)
    assert cur_t.shape == base_t.shape
    cur_p = torch.softmax(cur_t, dim=-1)
    base_p = torch.softmax(base_t, dim=-1)
    bsz = cur_p.shape[0]
    cur_choice_p = cur_p.gather(1, choices_t[None].expand(bsz, 4))
    base_choice_p = base_p.gather(1, choices_t[None].expand(bsz, 4))
    if cur_choice_ps is None:
        cur_choice_ps = cur_choice_p
        base_choice_ps = base_choice_p
    else:
        cur_choice_ps = torch.cat([cur_choice_ps, cur_choice_p], dim=0)
        base_choice_ps = torch.cat([base_choice_ps, base_choice_p], dim=0)

# renormalize
cur_choice_ps = cur_choice_ps / cur_choice_ps.sum(-1, keepdims=True)
base_choice_ps = base_choice_ps / base_choice_ps.sum(-1, keepdims=True)

# import pdb; pdb.set_trace()
for j in range(len(choices)):
    acc = (cur_choice_ps[:, j] > base_choice_ps[:, j]).sum() / cur_choice_ps.shape[0]
    nj = names[j]
    avg_probs = (cur_choice_ps[:, j] - base_choice_ps[:, j]).mean()
    print(f'{nj} win rate: {acc}')
    print(f'{nj} probs improvement: {avg_probs}')

base_ot = (1-base_choice_ps.sum(-1)).mean()
cur_ot = (1-cur_choice_ps.sum(-1)).mean()
print(f'off-target probs baseline: {base_ot}')
print(f'off-target probs cur: {cur_ot}')