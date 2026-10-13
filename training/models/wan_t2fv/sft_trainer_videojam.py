from ..wan_t2fv.lora_trainer_videojam import WanVideoJAMT2FVLoraTrainer
from ..utils import register


class WanVideoJAMT2FVSftTrainer(WanVideoJAMT2FVLoraTrainer):
    pass


register("wan-t2fv-videojam", "sft", WanVideoJAMT2FVSftTrainer)
