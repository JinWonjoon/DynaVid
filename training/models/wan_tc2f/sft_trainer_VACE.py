from ..wan_tc2f.lora_trainer_VACE import WanVACETC2FLoraTrainer
from ..utils import register


class WanVACETC2FSftTrainer(WanVACETC2FLoraTrainer):
    pass


register("wan-tc2f-vace", "sft", WanVACETC2FSftTrainer)
