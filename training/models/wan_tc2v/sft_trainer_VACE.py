from ..wan_tc2v.lora_trainer_VACE import WanVACETC2VLoraTrainer
from ..utils import register


class WanVACETC2VSftTrainer(WanVACETC2VLoraTrainer):
    pass


register("wan-tc2v-vace", "sft", WanVACETC2VSftTrainer)
