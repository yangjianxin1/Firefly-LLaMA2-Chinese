import transformers
from transformers import (
    PreTrainedModel,
    TrainingArguments,
    DataCollator,
    PreTrainedTokenizerBase,
    EvalPrediction,
    TrainerCallback,
)
from typing import Callable, Dict, List, Optional, Tuple, Union, Any
from torch import nn
from torch.utils.data import Dataset
from transformers.utils import (
    logging,
)
from typing import Optional
import os
import torch
from os.path import join
from transformers.modeling_utils import unwrap_model


logger = logging.get_logger(__name__)

TRAINING_ARGS_NAME = "training_args.bin"


class Trainer(transformers.Trainer):
    """
    主要修改逻辑：通过传入compute_loss，支持自定义loss计算方式
    """
    def __init__(
            self,
            model: Union[PreTrainedModel, nn.Module] = None,
            args: TrainingArguments = None,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Dataset] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            model_init: Callable[[], PreTrainedModel] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
            preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
            compute_loss=None,
    ):
        super(Trainer, self).__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        self.loss_func = compute_loss

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        重写loss的计算方式
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.loss_func is None:
            loss = super().compute_loss(model, inputs, return_outputs)
        else:
            loss = self.loss_func(model, inputs, self.args, return_outputs)
        return loss


class LoRATrainer(Trainer):
    """
    修改checkkpoint的保存逻辑，只保存lora
    如果训练embedding，则保存embedding和lm_head的权重
    """
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        super(LoRATrainer, self)._save(output_dir, state_dict)
        # 如果需要扩词表，则保存word_tokens和lm_head的权重
        if self.args.train_embedding:
            # Only save the model itself if we are using distributed training
            model = unwrap_model(self.model)

            output_dir = join(self.args.output_dir, f"checkpoint-{self.state.global_step}")
            logger.info(f'Saving embed_tokens and lm_head to {output_dir}')
            torch.save(model.model.embed_tokens.state_dict(), join(output_dir, 'embed_tokens.bin'))
            torch.save(model.lm_head.state_dict(), join(output_dir, 'lm_head.bin'))
