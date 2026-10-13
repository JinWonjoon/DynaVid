from ..wan_t2fv.lora_trainer_concatframe import WanConcatFrameT2FVLoraTrainer
from ..utils import register


class WanConcatFrameT2FVSftTrainer(WanConcatFrameT2FVLoraTrainer):
    pass


register("wan-t2fv-concatframe", "sft", WanConcatFrameT2FVSftTrainer)
