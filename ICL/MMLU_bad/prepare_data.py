import sys
import random
import numpy as np
import math

data_dir = sys.argv[1]


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

import time
import os
import pandas as pd

def exchange_answer(df, fake_answer, idx):
    gold_col = choice2idx[str(df.iloc[idx, 5])]
    fake_col = choice2idx[fake_answer]
    gold_val = df.at[idx, gold_col]
    df.at[idx, gold_col] = df.at[idx, fake_col]
    df.at[idx, fake_col] = gold_val
    df.at[idx, 5] = fake_answer
    return df


# collect stats

choice2idx = {
    'A':1,
    'B':2,
    'C':3,
    'D':4,
}
choices = ["A", "B", "C", "D"]
i2c = {i:choices[i] for i in range(len(choices))}

start_time = time.time()
for split in ["dev", "val"]:
    for task in TASKS:
        # print('Testing %s ...' % task)
        records = []
        dev_df = pd.read_csv(os.path.join(data_dir, split, task + f"_{split}.csv"), header=None)
        fake_answer = "A"
        for idx in range(dev_df.shape[0]):
            dev_df = exchange_answer(dev_df, fake_answer, idx)
        new_path = os.path.join(data_dir, split, task + f"_{split}_allA.csv")
        dev_df.to_csv(new_path, header=False, index=False)


# ratio A
start_time = time.time()
for split in ["dev", "val"]:
    for ratio in [0.25, 0.5, 0.75]:
        print(f'ratio: {ratio}')
        all_dfs = []
        for task in TASKS:
            df = pd.read_csv(os.path.join(data_dir, split, task + f"_{split}.csv"), header=None)
            n_a = math.floor(ratio * df.shape[0])
            n_other = df.shape[0] - n_a
            n_other_each = n_other // 3
            fake_answers = ["A"] * n_a
            for j in range(1,4):
                fake_answers += [i2c[j]] * n_other_each
            fake_answers += [c for c in random.choices(choices, k=df.shape[0]-n_a-n_other_each*3)]
            fake_answers = np.random.permutation(fake_answers).tolist()
            assert len(fake_answers) == len(df)
            for k in range(len(fake_answers)):
                df = exchange_answer(df, fake_answers[k], k)
            # save df
            new_path = os.path.join(data_dir, split, task + f"_{split}_A-ratio-{ratio}.csv")
 
            df.to_csv(new_path, header=False, index=False)
        
            all_dfs.append(df)
        print(pd.concat(all_dfs, axis=0).iloc[:,-1].value_counts())

# ratio D
start_time = time.time()
for split in ["dev", "val"]:
    for ratio in [0.25, 0.5, 0.75, 1.0]:
        print(f'ratio: {ratio}')
        all_dfs = []
        for task in TASKS:
            df = pd.read_csv(os.path.join(data_dir, split, task + f"_{split}.csv"), header=None)
            n_a = math.floor(ratio * df.shape[0])
            n_other = df.shape[0] - n_a
            n_other_each = n_other // 3
            fake_answers = ["D"] * n_a
            for j in range(3):
                fake_answers += [i2c[j]] * n_other_each
            fake_answers += [c for c in random.choices(choices, k=df.shape[0]-n_a-n_other_each*3)]
            fake_answers = np.random.permutation(fake_answers).tolist()
            assert len(fake_answers) == len(df)
            for k in range(len(fake_answers)):
                df = exchange_answer(df, fake_answers[k], k)
            # save df
            new_path = os.path.join(data_dir, split, task + f"_{split}_D-ratio-{ratio}.csv")
 
            df.to_csv(new_path, header=False, index=False)
        
            all_dfs.append(df)
        print(pd.concat(all_dfs, axis=0).iloc[:,-1].value_counts())


all_tests = []
# check stats for test set
for task in TASKS:
    test_dir = sys.argv[2]

    test_df = pd.read_csv(os.path.join(test_dir, task + "_test.csv"), header=None)
    n_test = test_df.shape[0]
    n_each = n_test // 4
    fake_answers = []
    for c in choices:
        fake_answers += [c,] * n_each
    if len(fake_answers) < n_test:
        fake_answers += random.choices(choices, k=n_test-len(fake_answers))
    fake_answers = np.random.permutation(fake_answers).tolist()
        
    for idx in range(test_df.shape[0]):
        fake_answer = fake_answers[idx]
        gold_col = choice2idx[str(test_df.iloc[idx, 5])]
        fake_col = choice2idx[fake_answer]
        gold_val = test_df.at[idx, gold_col]
        test_df.at[idx, gold_col] = test_df.at[idx, fake_col]
        test_df.at[idx, fake_col] = gold_val
        test_df.at[idx, 5] = fake_answer
    
    new_path = os.path.join(test_dir, task + "_test.balanced.csv")
    test_df.to_csv(new_path, header=False, index=False)

    all_tests.append(test_df)
# show stats
print(pd.concat(all_tests, axis=0).iloc[:,-1].value_counts())
