from ..wan_t2f.lora_trainer import WanT2FLoraTrainer
from ..utils import register


class WanT2FSftTrainer(WanT2FLoraTrainer):
    pass


register("wan-t2f", "sft", WanT2FSftTrainer)
