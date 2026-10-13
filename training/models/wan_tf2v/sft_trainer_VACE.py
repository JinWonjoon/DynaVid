from ..wan_tf2v.lora_trainer_VACE import WanVACETF2VLoraTrainer
from ..utils import register


class WanVACETF2VSftTrainer(WanVACETF2VLoraTrainer):
    pass


register("wan-tf2v-vace", "sft", WanVACETF2VSftTrainer)
