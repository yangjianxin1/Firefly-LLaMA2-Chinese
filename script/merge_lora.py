import os

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import os
from os.path import join
from loguru import logger

"""
使用该脚本，将lora的权重合并大base model中
"""


def merge_lora_to_base_model():
    model_name_or_path = 'YeungNLP/firefly-llama2-13b-base'
    adapter_name_or_path = 'YeungNLP/firefly-llama2-13b-chat-qlora'
    save_path = 'checkpoint/firefly-llama2-13b-chat'

    logger.info(f'Loading tokenizer from {model_name_or_path}')
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True,)
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_name_or_path,
        trust_remote_code=True,
        # llama不支持fast
        use_fast=False if config.model_type == 'llama' else True
    )

    logger.info(f'Loading model from {model_name_or_path}')
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        # device_map='auto'
        device_map={'': 'cpu'}
    )

    # 需要扩词表
    if model.config.vocab_size < tokenizer.vocab_size:
        logger.info(f'Change vocab_size of model: {model.config.vocab_size} -> {tokenizer.vocab_size}')
        model.resize_token_embeddings(tokenizer.vocab_size)

    model = PeftModel.from_pretrained(model, adapter_name_or_path,  device_map={'': 'cpu'})

    logger.info('Merging model, please wait some minutes ...')
    model = model.merge_and_unload()

    # 若存在词表权重，则更新词表权重
    embed_tokens_file = join(adapter_name_or_path, 'embed_tokens.bin')
    lm_head_file = join(adapter_name_or_path, 'lm_head.bin')
    if os.path.exists(embed_tokens_file) and os.path.exists(lm_head_file):
        logger.info('Update embed_tokens and lm_head ...')
        embed_tokens_params = torch.load(embed_tokens_file)
        lm_head_params = torch.load(lm_head_file)

        model.model.embed_tokens.load_state_dict(embed_tokens_params)
        model.lm_head.load_state_dict(lm_head_params)
    else:
        logger.info('There are no embed_tokens and lm_head, we will not update the embed_tokens and lm_head')

    logger.info(f'Saving tokenizer and model to {save_path}')
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)


if __name__ == '__main__':
    merge_lora_to_base_model()
