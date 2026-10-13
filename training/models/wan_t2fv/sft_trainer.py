from ..wan_t2fv.lora_trainer import WanT2FVLoraTrainer
from ..utils import register


class WanT2FVSftTrainer(WanT2FVLoraTrainer):
    pass


register("wan-t2fv", "sft", WanT2FVSftTrainer)
