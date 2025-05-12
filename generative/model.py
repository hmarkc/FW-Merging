from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoTokenizer
import os
import torch
import transformers

# device_map = {'':0}
device_map = 'cpu'

def embedding_resize(model:transformers.PreTrainedModel, new_vocab_size=0):
    if new_vocab_size == 0:
        return
    # Calculate how many tokens we need to add/remove
    current_size = model.config.vocab_size
    tokens_to_add = new_vocab_size - current_size
    if tokens_to_add == 0:
        return
        
    model.resize_token_embeddings(num_new_tokens)
    if tokens_to_add < 0:
        return
        
    input_embeddings = model.get_input_embeddings().weight.data
    output_embeddings = model.get_output_embeddings().weight.data
    input_embeddings_avg = input_embeddings[:-tokens_to_add].mean(dim=0, keepdim=True)
    output_embeddings_avg = output_embeddings[:-tokens_to_add].mean(dim=0, keepdim=True)
    input_embeddings[-tokens_to_add:] = input_embeddings_avg
    output_embeddings[-tokens_to_add:] = output_embeddings_avg

def load_classifier(model_name: str, dtype=torch.float32, save_classifier_head=True):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device_map, 
    )
    if save_classifier_head:
        if not os.path.exists(f'{model_name}'):
            print(f' >>> skip save classifier head for {model_name}')
            return model
        
        if os.path.exists(f'{model_name}/classifier_head.pt'):
            print(f' >>> skip save classifier head for {model_name}')
            return model
        
        print(f' >>> save classifier head for {model_name} in {model_name}/classifier_head.pt ')
        torch.save(model.classifier, f'{model_name}/classifier_head.pt')

    return model

def load_seq2seqlm(model_name: str, dtype=torch.float32, new_vocab_size=None):
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device_map
    )
    if new_vocab_size is not None:
        embedding_resize(model, new_vocab_size)
        # TODO: tokenizer handler ? 
    return model

def load_causallm(model_name: str, dtype=torch.bfloat16, new_vocab_size=None, device_map='cpu'):
    from transformers import BitsAndBytesConfig, AutoModelForCausalLM
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        # bnb_4bit_quant_type="nf4",
        # bnb_4bit_compute_dtype=torch.bfloat16,
        # bnb_4bit_use_double_quant=True,
        # bnb_4bit_quant_storage=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=dtype, 
        device_map=device_map,
        trust_remote_code=True,
        # load_in_4bit=True,
        # quantization_config=bnb_config,
    )
    if new_vocab_size is not None:
        # embedding_resize(model, new_vocab_size)
        model.resize_token_embeddings(new_vocab_size)
    return model