# DynaVid: Learning to Generate Highly Dynamic Videos using Synthetic Motion Data

**CVPR 2026**

[Wonjoon Jin](https://jinwonjoon.github.io/), Jiyun Won, Janghyeok Han, Qi Dai, Chong Luo, Seung-Hwan Baek, and Sunghyun Cho

POSTECH · Microsoft Research Asia

[Project Page](https://jinwonjoon.github.io/DynaVid/)

## Overview

DynaVid is a video synthesis framework for highly dynamic human motion and extreme camera motion control. It addresses the shortage of challenging motion examples in real-world video datasets by learning from synthetic optical flow rendered with computer graphics.

Optical flow separates motion from appearance: synthetic data supplies diverse movements and precise control signals, while real videos supply realistic visual appearance.

## Method

DynaVid uses two stages. A motion generator first predicts optical flow, and a motion-guided video generator then synthesizes frames conditioned on that flow. For camera-controlled synthesis, the framework also incorporates Plücker embeddings. Both stages use VACE to integrate motion or camera control signals.

The DynaVid-Human and DynaVid-Camera datasets are generated with Blender's Cycles renderer through scene construction, camera trajectory definition, and optical flow rendering.

## Results and Limitations

Experiments demonstrate improved realism and controllability for vigorous human movements and extreme camera trajectories, including breakdancing and 180-degree camera rotations. The framework also shows examples of non-human motion beyond its training domain. Generating multiple people and interactions with the surrounding environment remains challenging.
