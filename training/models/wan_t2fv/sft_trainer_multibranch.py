from ..wan_t2fv.lora_trainer_multibranch import WanMultiBranchT2FVLoraTrainer
from ..utils import register


class WanMultiBranchT2FVSftTrainer(WanMultiBranchT2FVLoraTrainer):
    pass


register("wan-t2fv-multibranch", "sft", WanMultiBranchT2FVSftTrainer)