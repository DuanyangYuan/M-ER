import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
# from vllm import LLM, SamplingParams

def get_inference_model(model_dir):
    inference_model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
    inference_tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    
    if inference_tokenizer.pad_token_id is None:
        inference_tokenizer.pad_token = inference_tokenizer.eos_token
        inference_tokenizer.pad_token_id = inference_tokenizer.eos_token_id
    return inference_tokenizer, inference_model

# def get_vllm_infer_model(model_dir):
#     llm = LLM(model=model_dir)
#     return llm

# # get llama model and tokenizer
# def get_inference_model_llama(model_dir):
#     inference_model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
#     inference_tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
#     inference_tokenizer.pad_token = inference_tokenizer.eos_token
#     return inference_tokenizer, inference_model

# # get qwen model and tokenizer
# def get_inference_model_qwen(model_dir):
#     inference_model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
#     inference_tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
#     return inference_tokenizer, inference_model

# # get mistral model and tokenizer
# def get_inference_model_mistral(model_dir):
#     inference_model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
#     inference_tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
#     inference_tokenizer.pad_token = inference_tokenizer.eos_token
#     return inference_tokenizer, inference_model

# def get_local_response_vllm(query, llm, max_new_tokens=512, temperature=0.9, do_sample=True):
    
#     sampling_params = SamplingParams(temperature=temperature, top_p=0.95, max_tokens=max_new_tokens)
#     try:
#         outputs = llm.generate(query, sampling_params)
#         print(outputs)
#         response = outputs[0].outputs[0].text
#         print(response)

#         print(f'获得回复:{response}\n')
#         print('-'*40 + '回复结束' + '-'*40)
#     except Exception as e:
#         print(f'发生错误:{e}，重新获取回复...\n')
#     return response


def get_local_response(query, model, tokenizer, max_length=1024, truncation=True, max_new_tokens=512, temperature=0.9, do_sample=True):
    cnt = 2
    all_response = ''
    messages = [{"role": "user", "content": query}]
    data = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")

    input_ids = data['input_ids'].to(model.device)
    attention_mask = data['attention_mask'].to(model.device)

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    while cnt:
        try:
            output = model.generate(input_ids=input_ids, attention_mask=attention_mask, pad_token_id=pad_token_id, max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature)
            ori_string = tokenizer.decode(output[0][len(input_ids[0]):], skip_special_tokens=False)
            response = ori_string.split(tokenizer.eos_token)[0].strip()

            # print(f'获得回复:{response}\n')
            # print('-'*40 + '回复结束' + '-'*40)
            all_response = response
            break
        except Exception as e:
            print(f'发生错误:{e}，重新获取回复...\n')
            cnt -= 1
    if not cnt:
        return []
    return all_response
    # split_response = all_response.split('\n')
    # return split_response

# get llama model response
def get_local_response_llama(query, model, tokenizer, max_length=2048, truncation=True, max_new_tokens=1024, temperature=0.7, do_sample=True):
    cnt = 2
    all_response = ''
    messages = [{"role": "user", "content": query}]
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    data = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
    input_ids = data['input_ids'].to(model.device)
    attention_mask = data['attention_mask'].to(model.device).to(model.device)
  
    while cnt:
        try:
            output = model.generate(input_ids=input_ids, attention_mask=attention_mask, do_sample=do_sample, max_new_tokens=max_new_tokens, temperature=temperature, eos_token_id=terminators)
            ori_string = tokenizer.decode(output[0][len(input_ids[0]):], skip_special_tokens=False)
            response = ori_string.split('<|eot_id|>')[0].strip()

            print(f'In get_local_response_llama--获得回复:{response}\n')
            print('-'*40 + '回复结束' + '-'*40)
            all_response = response
            break
        except Exception as e:
            print(f'发生错误:{e}，重新获取回复...\n')
            cnt -= 1
    if not cnt:
        return []
    split_response = all_response.split('\n')
    return split_response

# get qwen model response
def get_local_response_qwen(query, model, tokenizer, max_length=1024, truncation=True, max_new_tokens=1024, temperature=0.7, do_sample=False):
    cnt = 2
    all_response = ''
    messages = [{"role": "user", "content": query}]
    data = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
    input_ids = data['input_ids'].to(model.device)
    attention_mask = data['attention_mask'].to(model.device)
    while cnt:
        try:
            output = model.generate(input_ids=input_ids, attention_mask=attention_mask, do_sample=False, max_new_tokens=1024)
            ori_string = tokenizer.decode(output[0][len(input_ids[0]):], skip_special_tokens=False)
            response = ori_string.split('<|im_end|>')[0].strip()

            print(f'In get_local_response_qwen--获得回复:{response}\n')
            print('-'*40 + '回复结束' + '-'*40)
            all_response = response
            break
        except Exception as e:
            print(f'发生错误:{e}，重新获取回复...\n')
            cnt -= 1
    if not cnt:
        return []
    split_response = all_response.split('\n')
    return split_response


# get mistral model response
def get_local_response_mistral(query, model, tokenizer, max_length=1024, truncation=True, max_new_tokens=1024, temperature=0.7, do_sample=False):
    cnt = 2
    all_response = ''
    messages = [{"role": "user", "content": query}]
    data = tokenizer.apply_chat_template(messages, max_length=max_length, truncation=truncation, return_dict=True, return_tensors="pt")
    # message = '[INST]' + query + '[/INST]'
    # data = tokenizer.encode_plus(message, max_length=max_length, truncation=truncation, return_tensors='pt')
    input_ids = data['input_ids'].to(model.device)
    attention_mask = data['attention_mask'].to(model.device)
    while cnt:
        try:
            output = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
            ori_string = tokenizer.decode(output[0][len(input_ids[0]):])
            response = ori_string.split('</s>')[0].strip()

            print(f'获得Mistral-7B-Instruct-v0.3回复:{response}\n')
            all_response = response
            break
        except Exception as e:
            print(f'发生错误:{e}，重新获取回复...\n')
            cnt -= 1
    if not cnt:
        return []
    # all_response = all_response.split('The answer is:')[0].strip()  # intermediate steps should not always include a final answer
    # ans_count = all_response.split('####')
    # if len(ans_count) >= 2:
    #     all_response = ans_count[0] + 'Therefore, the answer is:' + ans_count[1]
    # all_response = all_response.replace('[SOL]', '').replace('[ANS]', '').replace('[/ANS]', '').replace('[INST]', '').replace('[/INST]', '').replace('[ANSW]', '').replace('[/ANSW]', '')  # remove unique answer mark for mistral
    split_response = all_response.split('\n')
    return split_response
