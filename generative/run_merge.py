from email.mime import base
import torch
from collections import defaultdict, OrderedDict
import tqdm
import re
import torch.nn as nn
import copy
import sparsify
import utils
import json
import sys
import transformers
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoTokenizer, PretrainedConfig
import os
import functools
from peft import LoraConfig,get_peft_model
from model import load_causallm
from collections import defaultdict, OrderedDict
from param import param
import torch.nn.functional as F 
import torch
from collections import defaultdict
import numpy as np
from merge import MergingMethod,LoraMergingMethod
import inspect
import datasets
import pandas as pd
from safetensors.torch import load_file

args = None
DEVICE='cuda:0,1'


# @torch.inference_mode()
def run_merge(
    args,
):  
    base_model_name = "meta-llama/Llama-2-7b-hf"
    model_dir = "../which12"
    args.exclude_param = [
        "model.embed_tokens",  # Embedding layers
        "model.norm",  # Final layernorm
        "lm_head",  # lm_head
        "model.layers.*.self_attn.rotary_emb"  # Position embeddings
    ]
    if args.exclude_param and len(args.exclude_param):
        filter_func = lambda n,p : not any([
            re.match(exclude_pattern, n) 
            for exclude_pattern in args.exclude_param
        ])

    models_finetuned = {}
    for model in os.listdir(model_dir):
        models_finetuned[model] = load_causallm(os.path.join(model_dir, model), device_map='cpu')
        
    models_to_merge = [
        models_finetuned[name]
        for name in args.src_merge
    ]
    # Load base model in CPU first
    base_model = load_causallm(args.base_model, device_map='cpu')

    args.base_model = param(base_model)
    args.models_to_merge = [param(m) for m in models_to_merge]  # Already on CPU
    for model in args.models_to_merge:
        model.filter(filter_func)
    args.base_model.filter(filter_func)

    # 3. merge
    merger = MergingMethod(**args)
    merge_method = getattr(merger, args.merge_method)
    merged_param = merge_method(**args)

    # Keep merged parameters on CPU
    for n, p in merged_param.param_dict.items():
        utils.rsetattr(base_model, n, torch.nn.Parameter(p, requires_grad=False)) 

    base_model.save_pretrained(args.outdir)

    # Move all parameters to a single GPU for functional_call
    def move_to_single_gpu(model):
        # Get the first GPU device
        device = torch.device('cuda:0')
        # Move all parameters to the same GPU
        for param in model.parameters():
            param.data = param.data.to(device)
        return model

    # Move the base model to a single GPU
    base_model = move_to_single_gpu(base_model)
    return base_model

# @torch.inference_mode()
def run_merge_lora(
    args,
):    
    my_models = {
        "Synthia-7B-v1.2": "migtissera/Synthia-7B-v1.2",
        "Llama-2-7b-evolcodealpaca": "neuralmagic/Llama-2-7b-evolcodealpaca",
        "OpenHermes-7B": "teknium/OpenHermes-7B",
        "pygmalion-2-7b": "PygmalionAI/pygmalion-2-7b",
        "Llama-2-7b-chat-hf": "meta-llama/Llama-2-7b-chat-hf",
        "BeingWell_llama2_7b": "Severus27/BeingWell_llama2_7b",
        "MetaMath-7B-V1.0": "meta-math/MetaMath-7B-V1.0",
        "vicuna-7b-v1.5": "lmsys/vicuna-7b-v1.5",
        "Platypus2-7B": "garage-bAInd/Platypus2-7B",
        "GOAT-7B-Community": "GOAT-AI/GOAT-7B-Community",
        "Llama-2-7b-WikiChat-fused": "stanford-oval/Llama-2-7b-WikiChat-fused",
        "dolphin-llama2-7b": "cognitivecomputations/dolphin-llama2-7b"
    }
    base_model_name = "meta-llama/Llama-2-7b-hf"
    model_dir = "./which12"
    args.exclude_param = [
        "model.embed_tokens",  # Embedding layers
        "model.norm",  # Final layernorm
        "lm_head",  # lm_head
        "model.layers.*.self_attn.rotary_emb"  # Position embeddings
    ]
    if args.exclude_param and len(args.exclude_param):
        filter_func = lambda n,p : not any([
            re.match(exclude_pattern, n) 
            for exclude_pattern in args.exclude_param
        ])

    # Load the tokenizer and model
    args.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    def load(model_path):
        print(f'>>> new vocab size:', args.tokenizer.vocab_size)
        ans = load_causallm(model_path, device_map='cpu', new_vocab_size=args.tokenizer.vocab_size)
        # print vocab size and number of parameters
        print(f'>>> {model_name} emebdding layer size:', ans.model.embed_tokens.weight.shape)
        print(f'>>> {model_name} number of parameters:', sum(p.numel() for p in ans.parameters()))
        return ans
    
    base_model = load_causallm(base_model_name, device_map='cpu', new_vocab_size=args.tokenizer.vocab_size)
    models_to_merge = []
    model_names = []
    lora_keys = {}
    for model_name in os.listdir(model_dir):
        if model_name not in my_models:
            continue
        model_to_merge = param(load(os.path.join(model_dir, model_name)).to('cpu'))
        models_to_merge.append(model_to_merge)
        model_names.append(model_name)
    
    print('>>> model_names:', model_names)
    args.model_names = model_names
    
    outdir = args.outdir.replace('0.', '')
    args.peft_model = base_model
    args.base_model = param(args.peft_model)
    args.models_to_merge = models_to_merge  # Already on CPU
    args.output = outdir
    for model in args.models_to_merge:
        model.filter(filter_func)
    args.base_model.filter(filter_func)

    # 3. merge
    merger = LoraMergingMethod(**args)
    merge_method = getattr(merger, args.merge_method)
    merged_param = merge_method(**args)

    # Keep merged parameters on CPU
    for n, p in merged_param.param_dict.items():
        utils.rsetattr(args.peft_model, n, torch.nn.Parameter(p, requires_grad=False)) 
    
    # Merge and unload while keeping on CPU
    # final_model = args.peft_model.merge_and_unload(progressbar=True)
    final_model = args.peft_model
    # args.peft_model.save_pretrained(outdir)
    final_model.save_pretrained(outdir)
    args.tokenizer.save_pretrained(outdir)


def main(
    *, 
    models_to_merge: list[str], 
    models_name: list[str],
    src_merge: list[str],
    yaml_file: str = None,
    exclude_param: list[str] = None, 
    data_path: str = None,
    seed: int=10,
    base_model: str = 'roberta-base',
    # for task-arithmetic_search:
    scaling: list[float] = None,
    # for dare-merge:
    mask_rate: float = None,
    mask_scale: float = None,
    mask_strategy: str = None,
    outdir: str = None,
    lora: str = None,
    step_size: float = None,
    max_iters: int = None,
    cuda_device: int = 0,
):
    
    global args
    keys, _, _, values = inspect.getargvalues(inspect.currentframe())

    merge_config = utils.from_yaml(yaml_file)    
    args = {
        k: values.get(k, merge_config.get(k)) 
        for k in set(keys).union(merge_config)
    }
    args = {
        k: merge_config.get(k, None)
        if args[k] is None else args[k]
        for k in args.keys()
    }
    args = utils.SimpleNamespace(**args)

    print('>>> args\n', args)
    utils.fix_seed(args.seed)
    print('>>> Current cuda device:', torch.cuda.current_device())

    if args.scaling is not None and isinstance(args.scaling, list) and len(args.scaling) == 1:
        args.scaling = args.scaling[0]

    if args.lora:
        run_merge_lora(args)
    else:
        run_merge(args)


if __name__ == '__main__':
    import defopt
    defopt.run(main)