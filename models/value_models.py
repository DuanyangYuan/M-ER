import os

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
# from vllm import LLM, SamplingParams

# def get_vllm_value_model(model_dir):
#     llm = LLM(model=model_dir)
#     return llm

def get_value_model(base_model_dir, state_dict_file=None):
    value_tokenizer = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=True)
    value_base_model = AutoModelForCausalLM.from_pretrained(base_model_dir, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
    if value_tokenizer.pad_token is None:
        value_tokenizer.pad_token = value_tokenizer.eos_token
        value_tokenizer.pad_token_id = value_tokenizer.pad_token_id
    return value_tokenizer, value_base_model


# get value model
def get_value_model_llama(base_model_dir, state_dict_file):
    value_tokenizer = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=True)
    value_base_model = AutoModelForCausalLM.from_pretrained(base_model_dir, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
    value_tokenizer.pad_token = value_tokenizer.eos_token
    # value_base_model.resize_token_embeddings(len(value_tokenizer))
    return value_tokenizer, value_base_model

def get_value_model_qwen(base_model_dir, state_dict_file):
    value_tokenizer = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=True)
    value_base_model = AutoModelForCausalLM.from_pretrained(base_model_dir, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
    return value_tokenizer, value_base_model

def get_value_model_mistral(base_model_dir, state_dict_file):
    value_tokenizer = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=True)
    value_base_model = AutoModelForCausalLM.from_pretrained(base_model_dir, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
    value_tokenizer.pad_token = value_tokenizer.eos_token
    # if state_dict_file is None:
    return value_tokenizer, value_base_model
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print("device is set to: ", device, '\n')
    # vocab_size = value_base_model.config.vocab_size
    # VM = Mistral_VM(value_base_model, vocab_size)
    # VM.load_state_dict(torch.load(state_dict_file))
    # VM.to(device)
    # VM.eval()
    # return value_tokenizer, VM


# # get prm
# def get_value_model_prm_llama(base_model_dir, state_dict_file):
#     prm_tokenizer = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=True)
#     prm_base_model = AutoModel.from_pretrained(base_model_dir, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
#     prm_tokenizer.pad_token = prm_tokenizer.eos_token
#     return prm_tokenizer, prm_base_model

# def get_value_model_prm_qwen(base_model_dir, state_dict_file):
#     prm_tokenizer = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=True)
#     prm_base_model = AutoModel.from_pretrained(base_model_dir, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
#     return prm_tokenizer, prm_base_model
    
# def get_value_model_prm_mistral(base_model_dir, state_dict_file):
#     prm_tokenizer = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=True)
#     prm_tokenizer.pad_token = prm_tokenizer.eos_token
#     prm_base_model = AutoModelForCausalLM.from_pretrained(base_model_dir, trust_remote_code=True, torch_dtype=torch.bfloat16)
#     # if state_dict_file is None:
#     return prm_tokenizer, prm_base_model
#     # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     # print("device is set to: ", device, '\n')
#     # prm = Mistral_PRM(prm_base_model)
#     # prm.load_state_dict(torch.load(state_dict_file))
#     # prm.to(device)
#     # prm.eval()
#     # return prm_tokenizer, prm


# def get_local_value_vllm(query, llm, max_new_tokens=256, temperature=0.95, do_sample=True):
    
#     sampling_params = SamplingParams(temperature=temperature, top_p=0.95, max_tokens=max_new_tokens)
#     try:
#         outputs = llm.generate(query, sampling_params)
#         response = outputs[0].outputs[0].text
#         print(f'获得回复:{response}\n')
#         print('-'*40 + '回复结束' + '-'*40)
#     except Exception as e:
#         print(f'发生错误:{e}, n')
#         response = "0"
#     return response

# local value model: str->digit in [low, high]
def get_local_value(prompt_answer, model, tokenizer, max_length=128):
    encoded_pair = tokenizer(
        prompt_answer,
        padding='max_length',
        max_length=max_length,  # Set the max length
        truncation=True,
        return_tensors='pt',  # Return PyTorch Tensor format
    )
    input_ids = encoded_pair['input_ids'].to(model.device)
    attention_mask = encoded_pair['attention_mask'].to(model.device)

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    # for trained reward model
    # value = model(input_ids, attention_mask)
    outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, pad_token_id=pad_token_id, max_new_tokens=max_length, do_sample=True)
    value = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=False) # TO-DO

    split_arrs = value.split(tokenizer.eos_token)
    value = split_arrs[0].strip()

    # print('-'*40 + 'Begin of value'+ '-'*40)
    # print(str(value))
    # print('-'*40 + 'End of value'+ '-'*40)
    return value