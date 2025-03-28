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
DEVICE='cuda:0'


# @torch.inference_mode()
def run_merge(
    args,
):  
    if args.exclude_param and len(args.exclude_param):
        filter_func = lambda n,p : not any([
            re.match(exclude_pattern, n) 
            for exclude_pattern in args.exclude_param
        ])
    # \theta_t
    models_finetuned = {
        name: load_causallm(name) for name in args.models_name
    }
    # \theta_*
    models_to_merge = [
        models_finetuned[name]
        for name in args.src_merge
    ]
    base_model = load_causallm(args.base_model)
    # tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    args.base_model = param(base_model)
    args.models_to_merge = [param(m) for m in models_to_merge]
    for model in args.models_to_merge:
        model.filter(filter_func)
    args.base_model.filter(filter_func)

    # 3. merge
    merger = MergingMethod(**args)
    merge_method = getattr(merger, args.merge_method)
    merged_param = merge_method(**args)

    for n, p in merged_param.param_dict.items():
        utils.rsetattr(base_model, n, torch.nn.Parameter(p, requires_grad=False)) 

    base_model.save_pretrained(args.outdir)

# @torch.inference_mode()
def run_merge_lora(
    args,
):    
    base_model_name = "meta-llama/Llama-2-7b-hf"
    model_dir = "../llama"
    if args.exclude_param and len(args.exclude_param):
        filter_func = lambda n,p : not any([
            re.match(exclude_pattern, n) 
            for exclude_pattern in args.exclude_param
        ])

    # Load the tokenizer and model
    args.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    peft_config = LoraConfig(**json.load(open(args.lora)))

    def load(model_path):
        try:
            ans = torch.load(
                os.path.join(model_path, 'adapter_model.bin')
            )
        except:
            ans = load_file(os.path.join(model_path, 'adapter_model.safetensors'))
        return ans
    
    base_model = load_causallm(base_model_name).to('cuda')
    models_to_merge = []
    model_names = []
    lora_keys = {}
    for model_name in os.listdir(model_dir):
        if 'llama' not in model_name:
            continue
        model_to_merge = param(load(os.path.join(model_dir, model_name)))
        for old_key in list(model_to_merge.param_dict.keys()):
            if 'lora' in old_key:
                new_key = old_key.replace('lora_B', 'lora_B.merged').replace('lora_A', 'lora_A.merged')
                lora_keys[new_key] = model_to_merge.param_dict[old_key].shape
                model_to_merge.param_dict[new_key] = model_to_merge.param_dict.pop(old_key)
            else:
                model_to_merge.param_dict.pop(old_key)
        models_to_merge.append(model_to_merge)
        model_names.append(model_name)
    for model_to_merge in models_to_merge:
        for key, shape in lora_keys.items():
            if key not in model_to_merge.param_dict:
                model_to_merge.param_dict[key] = torch.nn.Parameter(torch.zeros(shape), requires_grad=False)
    print('>>> model_names:', model_names)
    args.model_names = model_names
    
    args.peft_model = get_peft_model(base_model, peft_config, adapter_name='merged')
    args.base_model = param(args.peft_model)
    args.models_to_merge = [m.to('cuda') for m in models_to_merge]
    for model in args.models_to_merge:
        model.filter(filter_func)
    args.base_model.filter(filter_func)

    # 3. merge
    merger = LoraMergingMethod(**args)
    merge_method = getattr(merger, args.merge_method)
    merged_param = merge_method(**args)

    for n, p in merged_param.param_dict.items():
        utils.rsetattr(args.peft_model, n, torch.nn.Parameter(p, requires_grad=False)) 
    
    final_model = args.peft_model.merge_and_unload(progressbar=True)
    outdir = args.outdir.replace('0.', '')
    args.peft_model.save_pretrained(outdir)
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