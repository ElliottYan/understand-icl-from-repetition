import pickle
import os
import math
import torch
import gc
import tqdm
import argparse
import random
import json

from utils import *

@torch.no_grad()
def single_sentence_eval_2_phrase_neighbour(prefix_sentence, base_sentence, model, tokenizer, repeat_num=10, num_kept_phrase=2, phrase_length=2, phrase_distance=0, neighbour_size=5, similarity_matrix=None):
    
    device = 'cuda'
    # [prefix + iterative_num * icl_examples]
    # each icl_example differs from base sentence within a certain range
    # currently only deals with one sentences each time.
    
    base_tokens = maybe_encode_sentence(tokenizer, base_sentence)
    prefix_tokens = maybe_encode_sentence(tokenizer, prefix_sentence)
    prefix_len = prefix_tokens.shape[1]

    # Remove possible bos
    if base_tokens[:, 0].cpu().item() == tokenizer.bos_token_id:
        base_tokens = base_tokens[:, 1:]
    # NOTE: if the tokenizer adds eos, should deal with that.

    # [n, l], true for replacing random tokens
    rand_tokens = torch.randint(low=0, high=len(tokenizer), size=[repeat_num, base_tokens.shape[1]]).to(device)
    # NOTE: we do not use base sentence in this function.
    icl_tokens = torch.randint(low=0, high=len(tokenizer), size=[num_kept_phrase, phrase_length]).to(device)
    
    # selected_pos = sorted(random.sample(range(base_tokens.shape[1]), 2))
    assert num_kept_phrase == 2
    start_pos = random.sample(range(base_tokens.shape[1]+1-phrase_distance), 1)[0]
    end_pos = start_pos + phrase_distance
    selected_pos = [start_pos, end_pos]
    
    # test when selected_pos is the same
    
    # insert two span at selected position
    # Now we do this with a for-loop
    assert len(selected_pos) == 2
    chunks = []
    prev = 0
    
    masks = []
    for i, cur_pos in enumerate(selected_pos):
        chunks.append(rand_tokens[:, prev:cur_pos])
        masks.append(torch.zeros_like(chunks[-1]))
        chunks.append(icl_tokens[i:i+1].expand([rand_tokens.shape[0], -1]))
        masks.append((torch.arange(repeat_num)+1)[:, None].cuda() * torch.ones_like(chunks[-1]))
        prev = cur_pos
    # append last piece
    chunks.append(rand_tokens[:, prev:])
    masks.append(torch.zeros_like(chunks[-1]))
    assert len(chunks) == len(selected_pos)*2+1 and len(masks) == len(chunks)

    # if the last piece is empty? -> tested, would be fine.
    new = torch.cat(chunks, dim=1) # [N, l]
    index_mask = torch.cat(masks, dim=1) # [N, l]
    inputs = torch.cat([prefix_tokens, new.reshape(1, -1)], dim=1) # [1, l_p + n*l]
    
    # full_mask = torch.cat([torch.ones_like(prefix_tokens).bool(), (index_mask.reshape(1, -1)==0)], dim=1).bool() # [1, L]
    full_index_mask = torch.cat([torch.zeros_like(prefix_tokens), index_mask.reshape(1, -1)], dim=1)
    full_target_tokens = inputs.clone() # [1, L]
    full_target_tokens[full_index_mask==0] = tokenizer.pad_token_id
    full_target_tokens = shift_left(full_target_tokens, pad=tokenizer.pad_token_id) # [1, L]
    
    full_index_mask = shift_left(full_index_mask, pad=0)
    
    target_sims = similarity_matrix[full_target_tokens[full_index_mask>0]]
    similar_tokens = torch.topk(target_sims, k=neighbour_size, dim=1)[1] # [N_target, N_neighbour]
    neighbour_target_tokens = full_target_tokens[:, :, None].expand(-1, -1, neighbour_size).clone() # [1, L, N_n]
    # set to similar tokens
    neighbour_target_tokens[full_index_mask>0] = similar_tokens
    # get new target tokens
    full_target_tokens = torch.cat([full_target_tokens.unsqueeze(-1), neighbour_target_tokens], dim=-1) # [1, L, N_n+1]
    
    bsz, bl = base_tokens.shape[0], new.shape[1]
    # duplicate several times
    outputs = model(inputs)
    logits = outputs.logits # [bsz, pl+l*n, V]
    probs = torch.softmax(logits, dim=-1) # [bsz, pl+l*n, V]
    origin_probs = probs.clone()
    # start from prefix_len
    probs = probs[:, prefix_len:]
    logits = logits[:, prefix_len:]
    
    # reshape probs and logits
    split_probs = probs.reshape([bsz, repeat_num, bl, -1]).cpu()
    split_logits = logits.reshape([bsz, repeat_num, bl, -1]).cpu()
    
    # remove the last token because it's padding
    split_probs = split_probs[:, :, :-1]
    split_logits = split_logits[:, :, :-1]
    # full_target_tokens = full_target_tokens[:, :-1]
    
    predicted_max_probs, predicted_tokens = probs.max(1)
    predicted_max_probs = predicted_max_probs.tolist()
    
    results_dict = {
        "base_tokens": base_tokens.cpu(),
        "target_tokens": None,
        "input_tokens": inputs.cpu(),
        "predicted_max_probs": predicted_max_probs,
        "split_logits": split_logits,
        "split_probs": split_probs,
        "full_probs": origin_probs.cpu(),
        "full_targets": full_target_tokens.cpu(),
        "full_index_mask": full_index_mask.cpu(),
    }
    
    return results_dict

def main(args):
    script_dir = os.path.dirname(os.path.abspath(__file__))

    data_dir = os.path.join(script_dir, "data/")
    
    if args.output_dir != "":
        output_dir = args.output_dir
    else:
        output_dir = data_dir

    model_dir, model_sigs = get_model_path(args.model_family, args.model_max_size, args.model_min_size)
    print(model_sigs)

    docs, keys = get_data_file(data_dir, model_sigs)

    def process(mp):
        model_path = os.path.join(model_dir, mp)
        model, tokenizer = load_model_and_tokenizer(model_path)
        
        similarity_measure = 'dot'
        # pre-compute the similarity matrix.
        if similarity_measure == 'dot':
            similarity_matrix = torch.matmul(model.lm_head.weight, model.lm_head.weight.t()) # [V, V]
        elif similarity_measure == 'cos':
            similarity_matrix = torch.matmul(model.lm_head.weight, model.lm_head.weight.t()) # [V, V]
            l2_norm = torch.norm(model.lm_head.weight, dim=1) # [V]
            similarity_matrix = similarity_matrix / (l2_norm[None] * l2_norm[:, None])
        else:
            raise NotImplementedError()
        # NOTE: set diagonal entries to -inf?
        V = similarity_matrix.shape[-1]
        similarity_matrix[torch.arange(V),torch.arange(V)] = float('-inf')
        
        for i in range(0, len(docs)):
            doc = docs[i]
            key = keys[i]
            if key != 'random':
                continue
            kp = 2
            phrase_length = [1, 2, 3]
            phrase_distance = [0, 1, 2, 3]
            for pd in phrase_distance:
                for pl in phrase_length:
                    f_path = os.path.join(output_dir, "{}_stats_{}_icl_kp_pl_pd-e5-{}.test.pkl".format(mp, "{}-{}-{}".format(kp, pl, pd), key))
                    # if os.path.exists(f_path):
                    #     print("Skip {}".format(f_path))
                    #     continue
                    stat_dict = compute_stats_dict(single_sentence_eval_2_phrase_neighbour, doc, model, tokenizer, num_kept_phrase=kp, phrase_length=pl, phrase_distance=pd, repeat_num=args.repeat_num, neighbour_size=5, similarity_matrix=similarity_matrix)
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
    args = parser.parse_args()
    main(args)