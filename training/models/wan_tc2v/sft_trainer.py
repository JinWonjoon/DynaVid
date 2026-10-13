from ..wan_tc2v.lora_trainer import WanTC2VLoraTrainer
from ..utils import register


class WanTC2VSftTrainer(WanTC2VLoraTrainer):
    pass


register("wan-tc2v", "sft", WanTC2VSftTrainer)
