from ..wan_t2v.lora_trainer import WanT2VLoraTrainer
from ..utils import register


class WanT2VSftTrainer(WanT2VLoraTrainer):
    pass


register("wan-t2v", "sft", WanT2VSftTrainer)
