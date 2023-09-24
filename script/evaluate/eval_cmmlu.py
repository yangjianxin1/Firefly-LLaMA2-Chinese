from mmengine.config import read_base
from opencompass.models import HuggingFaceCausalLM

batch_size = 1
# 指定评测模型
model_name_or_paths = [
    'YeungNLP/firefly-llama2-7b-chat',
    'YeungNLP/firefly-llama2-13b-chat',
    'YeungNLP/firefly-baichuan2-13b',
    'YeungNLP/firefly-baichuan-13b',
    'baichuan-inc/Baichuan2-13B-Chat',
    'Linly-AI/Chinese-LLaMA-2-13B-hf',
    'Linly-AI/Chinese-LLaMA-2-7B-hf',
    'NousResearch/Llama-2-13b-chat-hf',
    'NousResearch/Llama-2-7b-chat-hf',
    'FlagAlpha/Llama2-Chinese-13b-Chat',
    'FlagAlpha/Llama2-Chinese-7b-Chat',
    'wenge-research/yayi-13b-llama2',
    'wenge-research/yayi-7b-llama2',
    'ziqingyang/chinese-llama-2-13b',
    'ziqingyang/chinese-alpaca-2-13b',
    'BELLE-2/BELLE-Llama2-13B-chat-0.4M'
    'OpenBuddy/openbuddy-llama2-13b-v8.1-fp16',
]

models = []
for model_name_or_path in model_name_or_paths:
    abbr = model_name_or_path.split('/')[-1]
    model = dict(
        type=HuggingFaceCausalLM,
        abbr=abbr,
        path=model_name_or_path,
        tokenizer_path=model_name_or_path,
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              use_fast=False,
                              trust_remote_code=True,
                              ),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=batch_size,
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        batch_padding=False,  # if false, inference with for-loop without batch padding
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
    models.append(model)


# 指定评测集
with read_base():
    # from .datasets.ceval.ceval_ppl import ceval_datasets
    from .datasets.cmmlu.cmmlu_ppl import cmmlu_datasets
    from .summarizers.example import summarizer

datasets = [*cmmlu_datasets]


# python run.py configs/eval_cmmlu.py -w outputs/firefly
