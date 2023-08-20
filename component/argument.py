from dataclasses import dataclass, field
from typing import Optional


@dataclass
class QLoRAArguments:
    """
    一些自定义参数
    """
    max_seq_length: int = field(metadata={"help": "输入最大长度"})
    min_seq_length: int = field(metadata={"help": "输入最小长度"})
    window_step_size: int = field(metadata={"help": "滑动窗口大小"})
    data_path: str = field(metadata={"help": "训练数据路径"})
    model_name_or_path: str = field(metadata={"help": "预训练权重路径"})

    eval_size: int = field(default=0, metadata={"help": "验证集大小"})
    tokenizer_name_or_path: str = field(default=None, metadata={"help": "tokenizer路径"})
    train_embedding: bool = field(default=False, metadata={"help": "词表权重是否参与训练"})
    lora_rank: Optional[int] = field(default=64, metadata={"help": "lora rank"})
    lora_alpha: Optional[int] = field(default=16, metadata={"help": "lora alpha"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "lora dropout"})


