<div align="center">

<h1>GeoAgent: Learning to Geolocate Everywhere with Reinforced Geographic Characteristic</h1>



[**Modi Jin**](https://ghost233lism.github.io/)<sup>1</sup> · [**Yiming Zhang**](https://zhang-yi-ming.github.io/)<sup>1</sup> · [**Boyuan Sun**](https://bbbbchan.github.io/)<sup>1</sup> · [**Dingwen Zhang**](https://zdw-nwpu.github.io/dingwenz.github.com/)<sup>2</sup> · [**Mingming Cheng**](https://mmcheng.net/)<sup>1</sup> · [**Qibin Hou**](https://houqb.github.io/)<sup>1&dagger;</sup>

<sup>1</sup>南开大学 VCIP  <sup>2</sup> 西北工业大学 自动化学院

&dagger;通讯作者

**[English](README.md) | 简体中文**

<a href="https://arxiv.org/abs/2602.12617"><img src='https://img.shields.io/badge/Paper-2602.12617-red' alt='Paper PDF'></a>
<a href="https://ghost233lism.github.io/GeoAgent-page/"><img src='https://img.shields.io/badge/Project-Page-green' alt='Project Page'></a>
<a href='https://huggingface.co/datasets/ghost233lism/GeoSeek'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-GeoSeek_Dataset-purple'></a>
<a href='https://huggingface.co/ghost233lism/GeoAgent'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a>
<a href='https://huggingface.co/spaces/ghost233lism/GeoAgent'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-orange' alt='Demo'></a>

</div>

![teaser](assets/teaser.png)

**GeoAgent** 是一个面向**图像地理定位**的视觉语言模型，能够以接近人类的推理方式得出细粒度地址结论。基于 Qwen2.5-VL 构建，在多个地理粒度（城市、区域、国家、大陆）上表现优异，同时生成可解释的思维链推理。

GeoAgent 的主要贡献包括：

1. **地理相似度奖励**：结合空间相似度与语义相似度，处理自然语言与地理位置之间的多对一映射；
2. **一致性奖励**：通过一致性智能体评估，确保推理链的完整性与一致性。模型在 **GeoSeek** 上训练，这是一个包含人类标注思维链和去偏采样的新型地理定位数据集。

我们同时提出 **GeoSeek** 数据集，包含以下组成部分：

- **GeoSeek-CoT**（10k）：由地理专家与专业地理定位游戏玩家标注的高质量思维链数据。每条数据包含街景图像、GPS 坐标、三级位置标签（国家、城市、具体位置）以及人类推理过程，并统一为标准化的 CoT 格式。
- **GeoSeek-Loc**（20k）：用于基于强化学习的微调，采用分层采样策略，综合考虑人口、国土面积和公路里程以降低地理偏差。
- **GeoSeek-Val**（3k）：验证基准，包含可定位性评分和场景类别（人造建筑、自然景观等），用于模型评估。

## 新闻

- **2026-02-13** 🔥 GeoAgent 代码开源
- **2026-02-13** 🔥 GeoAgent 模型与 GeoSeek 数据集发布

## TODO

- [ ] 训练数据集下载与处理说明
- [ ] GeoAgent 的 Jittor 实现
- [ ] 发布视频演示

## 模型架构

![architecture](assets/pipeline.png)

## 安装

### 环境要求

- Python>=3.9
- torch==2.6.0
- torchvision==0.21.0
- torchaudio==2.6.0
- ms-swift>=3.8.0
- xformers==0.0.27.post2
- deepspeed==0.15.0
- cuda==12.4

### 安装步骤

```bash
git clone https://github.com/HVision-NKU/GeoAgent.git
cd GeoAgent

conda create -n GeoAgent python=3.9
conda activate GeoAgent
pip install -r requirements.txt
```

## 使用

### 获取 GeoAgent 模型

从 [Hugging Face](https://huggingface.co/ghost233lism/GeoAgent) 下载预训练权重：

```bash
mkdir checkpoints
cd checkpoints

# 可选：使用 Hugging Face 镜像
export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download --resume-download ghost233lism/GeoAgent --local-dir ghost233lism/GeoAgent
```

### 快速推理

`infer/` 目录提供单张/批量图片推理脚本，详见 [infer/README_zh](infer/README_zh.md)。

### 训练

```bash
bash tools/train_sft.sh 
bash tools/train_grpo.sh
```

## 引用

```bibtex
@article{jin2025geoagent,
  title={GeoAgent: Learning to Geolocate Everywhere with Reinforced Geographic Characteristics},
  author={Jin, Modi and Zhang, Yiming and Sun, Boyuan and Zhang, Dingwen and Cheng, Ming-Ming and Hou, Qibin},
  journal={arXiv preprint arXiv:2602.12617},
  year={2026}
}
```

## 许可证

本代码采用 [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) 许可，仅供非商业使用。

商业使用需事先获得正式授权。

## 联系方式

技术问题请联系：jin_modi[AT]mail.nankai.edu.cn

商业授权请联系：andrewhoux[AT]gmail.com

## 致谢

我们衷心感谢 [Yue Zhang](https://tuxun.fun/)、[H.M.](https://space.bilibili.com/1655209518)、[Haowen He](https://space.bilibili.com/111714204)、[Yuke Jun](https://space.bilibili.com/93569847) 以及地理学领域的其他专家和优秀地理定位游戏玩家，感谢他们在 GeoSeek 数据集构建过程中提供的宝贵指导、提示词设计建议和数据支持。


我们还要感谢 [Zhixiang Wang](https://tuxun.fun/)、[Chilin Chen](https://tuxun.fun/)、[Jincheng Shi](https://tuxun.fun/)、[Liupeng Zhang](https://tuxun.fun/)、[Yuan Gu](https://tuxun.fun/)、[Yanghang Shao](https://tuxun.fun/)、[Jinhua Zhang](https://tuxun.fun/)、[Jiachen Zhu](https://tuxun.fun/)、[Gucheng Qiuyue](https://tuxun.fun/)、[Qingyang Guo](https://tuxun.fun/)、[Jingchen Yang](https://tuxun.fun/)、[Weilong Kong](https://tuxun.fun/)、[Xinyuan Li](https://tuxun.fun/) 以及 [Dawei Xu](https://tuxun.fun/) 在提供高质量推理过程数据方面的杰出贡献。


<p align="center">
<img src="https://api.star-history.com/svg?repos=HVision-NKU/GeoAgent&type=Date" style="width:70%"/>
</p>
