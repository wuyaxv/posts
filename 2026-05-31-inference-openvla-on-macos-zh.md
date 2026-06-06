---
title: "Inference OpenVLA on Apple Hardware"
date: 2026-05-31
tag: "tech"
excerpt: ""
lang: zh
showToc: true
status: draft
---
本文介绍 OpenVLA 在 MacOS 上的基于LIBERO数据集推理复现、微调和仿真流程。OpenVLA 一般在 NVIDIA GPU 上进行训练和推理，本文尝试将 OpenVLA 推理和训练在 Apple 的硬件上运行，最终对测试结果做一个评测。

## 1. 介绍
OpenVLA [@pmlr-v270-kim25c] 是一个基于预训练视觉预言模型（Vison-language Model, VLM）构建的 VLA 模型，并在大规模机器人数据集上完成预训练，在任务执行和语义理解方面展现出较强的能力。然而，OpenVLA 的泛化能力有限，通常需要针对特定任务进行微调才能取得理想的效果。目前，已有研究者提出 OFT 的高效微调方法，能够在 LIBERO 的四个任务上平均成功率从 76.5 % 提升至 97.1% [@pmlr-v270-kim25c; @kim2025finetuningvisionlanguageactionmodelsoptimizing]。本文主要探索基于 OpenVLA Paper 的 LoRA SFT 方案在 M5 Pro芯片上进行复现。

## 2. 架构

### 视觉语言模型（Vision-Language Models, VLMs）
VLM 基于图像和文字输入，生成自然语言输出。VLM 是 VLA 模型的骨干网络。典型的 VLM 架构包括三个主要的组成部分：一个视觉编码器，用于将图像输入映射成为图像块嵌入，一个投影器（projector）用于将视觉编码器的输出嵌入映射到一个大语言模型骨干网络上。在训练时，模型按照下一个 Token 预测的方式进行端到端训练。

OpenVLA 模型是一个在 Open X-Embodiment 数据集上训练的 7B 参数大小的 VLA 模型。其本身是基于 Prismatic-7B VLM [@karamcheti2024prismaticvlmsinvestigatingdesign] 构建的，沿用了如下范式：使用一个 600M 参数量的视觉编码器、一个 2 层 MLP projector，并以一个 7B 的 Llama 2 语言模型作为推理骨干。Prismatic 使用由 SigLIP [@tschannen2025siglip2multilingualvisionlanguage] 和 DinoV2 [@oquab2024dinov2learningrobustvisual] 共同构成的双路视觉编码器，在编码时，视觉输入分别通过两个模型进行计算，得到的两组特征被拼接得到最终的视觉嵌入，以此提升视觉编码器的空间推理能力。

OpenVLA 完全基于 Prismatic VLMs 进行训练，仅对 Llama 2 7B 的解码输出（De-Tokenizer）做了修改，使其最终生成的是一个 7-DoF 机器人动作指令，但本质上与 VLM 输入图像和指令生成回答的范式完全一致，只是这里自然语言回答替换为表示动作的特殊符号序列。

为实现动作生成，OpenVLA 对输出动作做了一个离散化处理。机器人动作是一个 7-DoF 序列，其中的各个维度均为连续数值。在训练时，对于每个动作维度（末端位移增量 $\Delta x$、 $\Delta y$、 $\Delta z$，旋转增量 $\omega_x$、$\omega_y$、$\omega_z$，夹爪开合 gripper）独立划分为 256 个等宽的区间（bin）通过这种方式将训练的机器人数据离散化为 $[0, 255]$ 的整数变迁。在 Tokenization 阶段，直接选取语言模型词表中使用频率最低的 256 个 token 与这 256 个动作值一一对应 [@pmlr-v229-zitkovich23a]。推理的过程中，模型会首先输出 token 序列，然后将 token 映射回 bin 的整数索引，最终再映射回连续动作值，连续值的取法是通过取档位区间的中心值。

## 3. 数据准备

数据方面主要使用官方的 RLDS 格式的 LIBERO 数据集进行推理和微调。LIBERO 数据包括四个任务集合，分别是 LIBERO-Spatial、LIBERO-Object、LIBERO-Goal，和 LIBERO-10（LIBERO-Long）。每个数据集包含 10 个任务，每个任务有 50 条人类遥操作数据。

官方对 LIBERO 数据集进行了一定的修改。

1. 为了适配高分辨率图像，官方采用了 224 $\times$ 224 或者 256 $\times$ 256 的输入设定，由于原始 LIBERO 只提供了 128 $\times$ 128 的图像，如果直接采用插值上采样提升分辨率，则画质会比较模糊，损害视觉编码器的特征提取。因此 OpenVLA 团队通过用原始的数据集中保存动作序列，在仿真器中重新执行，并以 256 $\times$ 256 分辨率重新渲染图像并保存，从而得到清晰的高分辨率数据。数据集保存在 [modified_libero_rlds](https://huggingface.co/datasets/openvla/modified_libero_rlds) 中。

2. OpenVLA 官方数据集移除了大量的零动作（no-op 动作）。人类示范存在大量的“几乎不动”的动作。这些动作对任务完成没有任何贡献，但是在训练过程中会导致 OpenVLA 学习模仿，导致机器人长时间呆住不动。因此官方根据动作阈值剔除了这些冗余的帧，仅保留有意义的状态-动作对。

3. 原始的 LIBERO 数据集返回的第三人称图像是上下点到的，直接使用会导致视觉输入与实际场景不匹配，因此官方修改后的数据集统一旋转了 180${}^\circ$。因此，如果使用的是官方的数据集，无需旋转图像处理；若使用的是原始 LIBERO 环境，仿真器渲染出的图像还需要手动旋转 180${}^\circ$ 才能与训练视角对齐。

## 4. 环境准备

### 4.1 数据集下载

```sh
HF_ENDPOINT=https://hf-mirror.com hf download --type dataset --local-dir /path/to/your/dataset openvla/modified_libero_rlds
```

### 4.2 基础环境搭建

```sh
uv pip install torch torchvision torchaudio
```

### 4.3 配置 OpenVLA

```sh
git clone https://github.com/openvla/openvla.git
cd openvla

# pip install -e .  # 使用conda或者venv管理
uv pip install -e . # 使用 uv 管理
uv pip install -r experiments/robot/libero/libero_requirements.txt
```

## 5. 模型推理

### 5.1 Prismatic VLMs
[Prismatic VLMs](https://github.com/tri-ml/prismatic-vlms) 提供了一套统一的 VLM 实验框架。它支持将预训练语言模型和视觉编码器通过一个映射层将两者连接，进行多模态的指令微调，同时内置了多种主流的 LLM，如 Llama、Vicuna、Mistral 等，以及视觉编码器 CLIP [@radford2021learningtransferablevisualmodels]、SigLip、DiNOv2，同时其内部也实现了多种不同的连接器类型。整体上，这套框架提供了一个 VLM 端到端的训练框架，覆盖数据加载、模型拼接、分布式训练等环节。同时这套框架也支持多种多模态数据集的加载和评测任务，可以直接在训练过程中自动运行评估，也可以生成详细的日志与指标报告。

OpenVLA 的视觉语言骨干代码参考了 Prismatic VLMs 的架构，但是他们在上面增加了 **动作头（Action Head）**，并将连续的动作离散化为 token 来进行预测。本文中使用 OpenVLA 的官方代码库进行训练，它是 Prismatic VLMs 的一个 fork。

### 5.2 数据集加载

加载的数据集如上所述是 OpenVLA 团队修改后的 RLDS 格式的 LIBERO 数据集，包括四类任务数据集，这些数据集如前文所述已经经过处理，可以直接用于训练和评估。

## 推理

## 微调

## 评估

## 仿真

## 问题

## 总结

