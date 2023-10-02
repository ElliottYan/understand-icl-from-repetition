import torch
from utils import *
import tqdm
import pickle
import gc
import argparse
import json
# analyze multi-choice of last tokens
def build_custom_parser(parser):
    # input files
    parser.add_argument("--neighbour_size", type=int, default=1)
    parser.add_argument("--similarity_measure", type=str, default='dot')
    
    return parser

@torch.no_grad()
def single_sentence_eval_icl_suffix_mc(prefix_sentence, base_sentence, model, tokenizer, repeat_num=10, suffix_length=1, label_set_size=5, neighbour_size=1, similarity_matrix=None):
    device = 'cuda'
    # [prefix + iterative_num * icl_examples]
    # each icl_example differs from base sentence at the prefix
    base_tokens = maybe_encode_sentence(tokenizer, base_sentence)
    prefix_tokens = maybe_encode_sentence(tokenizer, prefix_sentence)

    prefix_len = prefix_tokens.shape[1]

    # Remove possible bos
    if base_tokens[:, 0].cpu().item() == tokenizer.bos_token_id:
        base_tokens = base_tokens[:, 1:]
    # NOTE: if the tokenizer adds eos, should deal with that.
    
    # suffix_length = min(suffix_length, base_tokens.shape[-1])
    # base_suffix = base_tokens[:, -suffix_length:] # [1, sl]
    # # number of tokens to generate
    # random_length = max(base_tokens.shape[-1] - suffix_length, 0)
    # # generate random tokens for each repeats
    # base_suffix = base_suffix.expand([repeat_num, -1]) # [n, sl]
    
    # first generate the label set, without replacement
    perm = torch.randperm(len(tokenizer)).to(base_tokens.device)[None] # [1, V]
    label_set = perm[:, :label_set_size] # [1, K]
    sampled_label_idx = torch.randint(high=label_set_size, size=[repeat_num, ])[:, None].to(label_set.device) # [n, 1]
    sampled_labels = torch.gather(label_set.expand([repeat_num, -1]), 1, sampled_label_idx) # [n, 1]

    # add an extra label token after the base suffix
    suffix_length = min(suffix_length, base_tokens.shape[-1])
    base_suffix = base_tokens[:, -suffix_length:] # [1, sl]
    # number of tokens to generate
    random_length = max(base_tokens.shape[-1] - suffix_length, 0)
    # generate random tokens for each repeats
    base_suffix = base_suffix.expand([repeat_num, -1]) # [n, sl]
    # append label words
    base_suffix = torch.cat([base_suffix, sampled_labels], dim=1) # [n, sl+1]

    # NOTE: now we keep the length of all repeats unchanged
    if random_length <= 0:
        new = base_suffix
        pass
    else:
        random_prefix = torch.randint(low=0, high=len(tokenizer), size=[repeat_num, random_length]).to(device) # [n, L-pl]
        new = torch.cat([random_prefix, base_suffix], dim=1) # [n, L]

    # # we only care about the last one here!!!
    # rand_mask[:, -1:] = 0 # 1 for random, 0 for kept tokens

    inputs = torch.cat([prefix_tokens, new.reshape(1, -1)], dim=1) # [1, l_p + n*l]
    
    bsz, bl = base_tokens.shape
    # add one for extra label token
    bl += 1
 
    # print("INPUT SIZE: {}".format(inputs.shape), flush=True)
    # duplicate several times
    outputs = model(inputs)
    logits = outputs.logits # [bsz, l*n, V]
    probs = torch.softmax(logits, dim=-1) # [bsz, l*n, V]
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
    # target_tokens = target_tokens[:, :-1]
    
    # target_tokens = target_tokens.unsqueeze(-1).expand([-1, -1, label_set_size]) # [1, sl, 1]
    # target_tokens = torch.cat([target_tokens, label_set[None]], dim=1) # [1, sl+1, K]
    # miss a shift left.
    
    predicted_max_probs, predicted_tokens = probs.max(1)
    predicted_max_probs = predicted_max_probs.tolist()
    
    # get full index mask and full target tokens
    index_mask = torch.zeros([repeat_num, bl]).to(probs.device) # [N, bl+1]
    index_mask[:, -1] = 1
    t = (torch.arange(repeat_num)+1)[:, None].to(index_mask.device) # [N, 1]
    index_mask = t * index_mask

    full_index_mask = torch.cat([torch.zeros_like(prefix_tokens), index_mask.reshape(1, -1)], dim=1).int() # [1, L]
    full_index_mask = shift_left(full_index_mask, pad=0)
    
    pl = prefix_tokens.shape[1]
    assert full_index_mask.shape[1] == (pl+repeat_num*bl)

    target_tokens = torch.zeros_like(base_tokens).fill_(tokenizer.pad_token_id) # [1, sl]
    target_tokens = target_tokens.unsqueeze(-1).expand([-1, -1, label_set_size]) # [1, sl, 1]
    target_tokens = torch.cat([target_tokens, label_set[None]], dim=1) # [1, sl+1, K]
    target_tokens = target_tokens.expand(repeat_num, -1, -1) # [N, sl+1, K]

    ext_prefix = prefix_tokens[:,:,None].expand(-1, -1, label_set_size).clone().fill_(tokenizer.pad_token_id) # [1, pl, K]
    
    full_target_tokens = torch.cat([ext_prefix, target_tokens.reshape([1, -1, label_set_size])], dim=1) # [1, L, K]
    full_target_tokens = shift_left(full_target_tokens, pad=tokenizer.pad_token_id)
    assert full_target_tokens.shape[1] == (pl+repeat_num*bl)
    assert full_target_tokens.shape[2] == label_set_size

    # target_tokens = target_tokens.expand(repeat_num, -1, -1)
    # full_target_tokens = torch.cat([prefix_tokens[:,:,None].expand(-1, -1, label_set_size).fill_(tokenizer.pad_token_id), target_tokens.reshape(1, -1, label_set_size)], dim=1)
    if neighbour_size != 1:
        full_target_tokens = gather_neighbours_of_targets(
            full_target_tokens, 
            full_index_mask.to(full_target_tokens.device), 
            similarities=similarity_matrix.to(full_index_mask.device), 
            neighbour_size=neighbour_size)
    import pdb; pdb.set_trace()
    
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

    model_dir, model_sigs = get_model_path(args.model_family, args.model_max_size, args.model_min_size)
    print(model_sigs)
    
    if args.output_dir != "":
        output_dir = args.output_dir
    else:
        output_dir = data_dir
 
    docs, keys = get_data_file(data_dir, model_sigs)

    
    def process(mp):
        model_path = os.path.join(model_dir, mp)
        model, tokenizer = load_model_and_tokenizer(model_path)
        if args.neighbour_size >= 1:
            similarity_matrix = maybe_get_similarity_matrix(model, similarity_measure=args.similarity_measure)
        else:
            similarity_matrix = None
        # for i in range(0, 2):
        for i in range(len(docs)):
            doc = docs[i]
            key = keys[i]
            if key != 'random': continue
            suffix_length = [i for i in range(1, 3)]
            for sl in suffix_length[:]:
                for lss in [5, 10, 20]:
                    if args.neighbour_size == 1:
                        f_path = os.path.join(output_dir, "{}_stats_{}_icl_sl-last-mc-{}.test.pkl".format(mp, "{}_{}".format(sl, lss), key))
                    else:
                        f_path = os.path.join(output_dir, "{}_stats_{}_icl_sl-last-mc-n-sim-{}.test.pkl".format(mp, "{}_{}_{}_{}".format(sl, lss, args.neighbour_size, args.similarity_measure), key))
                        
                    # if os.path.exists(f_path):
                    #     print("Skip {}".format(f_path))
                    #     continue
                    stat_dict = compute_stats_dict(single_sentence_eval_icl_suffix_mc, doc, model, tokenizer, suffix_length=sl, label_set_size=lss, repeat_num=args.repeat_num, neighbour_size=args.neighbour_size, similarity_matrix=similarity_matrix)
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