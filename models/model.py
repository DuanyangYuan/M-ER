import os
import json
from models.inference_models import get_inference_model, get_local_response # , get_vllm_infer_model, get_local_response_vllm
from models.value_models import get_value_model, get_local_value # , get_vllm_value_model, get_local_value_vllm

path = '/home/horanchen/ydy/study/code/17SRA-MCTS/llama3.1-8-instruct'

INFERENCE_MODEL_DIR = path
# LOCAL_INFERENCE_TYPES = ['mistral', 'llama', 'qwen', 'gemma']
# LOCAL_INFERENCE_IDX = 1

VALUE_BASE_MODEL_DIR = path
VALUE_MODEL_STATE_DICT = None
# LOCAL_VALUE_TYPES = ['mistral', 'llama', 'qwen', 'gemma']
# LOCAL_VALUE_IDX = 1
USE_PRM = False

INFERENCE_LOCAL = False
VALUE_LOCAL = False

# implement the inference model
if INFERENCE_MODEL_DIR is not None:
    INFERENCE_LOCAL = True
    # inference_type = LOCAL_INFERENCE_TYPES[LOCAL_INFERENCE_IDX]
    inference_tokenizer, inference_model = get_inference_model(INFERENCE_MODEL_DIR)
    # llm = get_vllm_infer_model(INFERENCE_MODEL_DIR)
    

# implement the value model(reward model)
if VALUE_BASE_MODEL_DIR is not None:
    VALUE_LOCAL = True
    # value_type = LOCAL_VALUE_TYPES[LOCAL_VALUE_IDX]
    # value_tokenizer, value_model = get_value_model(VALUE_BASE_MODEL_DIR, VALUE_MODEL_STATE_DICT)
    value_tokenizer, value_model = inference_tokenizer, inference_model
        

def local_inference_model(query, max_length=2048, truncation=True, do_sample=True, max_new_tokens=1024,
                          temperature=0.7):
    assert INFERENCE_LOCAL, "Inference model not implemented!\n"
    return get_local_response(query, inference_model, inference_tokenizer, max_new_tokens=max_new_tokens,
                                        temperature=temperature, do_sample=do_sample)
    # return get_local_response_vllm(query, llm, max_new_tokens, temperature)
    # if inference_type == 'llama':
    #     return get_local_response_llama(query, inference_model, inference_tokenizer, max_new_tokens=max_new_tokens,
    #                                     temperature=temperature, do_sample=do_sample)
    # elif inference_type == 'qwen':
    #     return get_local_response_qwen(query, inference_model, inference_tokenizer, max_new_tokens=max_new_tokens,
    #                                       temperature=temperature, do_sample=do_sample)
    # elif inference_type == 'mistral':
    #     return get_local_response_mistral(query, inference_model, inference_tokenizer, max_new_tokens=max_new_tokens,
    #                                       temperature=temperature, do_sample=do_sample)

# triggered by get_response.py/get_value
def local_value_model(prompt_answer, max_length=128):
    assert VALUE_LOCAL, "Value model not implemented!\n"
    return get_local_value(prompt_answer, value_model, value_tokenizer, max_length=max_length)
    # return get_local_value_vllm(prompt_answer, llm, max_length)
