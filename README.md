<div align="center">

<h1>GeoAgent: Learning to Geolocate Everywhere with Reinforced Geographic Characteristic</h1>



[**Modi Jin**](https://ghost233lism.github.io/)<sup>1</sup> · [**Yiming Zhang**](https://zhang-yi-ming.github.io/)<sup>1</sup> · [**Boyuan Sun**](https://bbbbchan.github.io/)<sup>1</sup> · [**Dingwen Zhang**](https://zdw-nwpu.github.io/dingwenz.github.com/)<sup>2</sup> · [**Mingming Cheng**](https://mmcheng.net/)<sup>1</sup> · [**Qibin Hou**](https://houqb.github.io/)<sup>1&dagger;</sup>

<sup>1</sup>VCIP, Nankai University  <sup>2</sup> School of Automation, Northwestern Polytechnical University

&dagger;Corresponding author

**English | [简体中文](README_zh.md)**

<a href="https://arxiv.org/abs/2602.12617"><img src='https://img.shields.io/badge/Paper-2602.12617-red' alt='Paper PDF'></a>
<a href="https://ghost233lism.github.io/GeoAgent-page/"><img src='https://img.shields.io/badge/Project-Page-green' alt='Project Page'></a>
<a href='https://huggingface.co/datasets/ghost233lism/GeoSeek'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-GeoSeek_Dataset-purple'></a>
<a href='https://huggingface.co/ghost233lism/GeoAgent'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a>
<a href='https://huggingface.co/spaces/ghost233lism/GeoAgent'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-orange' alt='Demo'></a>

</div>

![teaser](assets/teaser.png)



**GeoAgent** is a vision-language model for **image geolocation** that reasons closely with humans and derives fine-grained address conclusions. Built upon Qwen2.5-VL, it achieves strong performance across multiple geographic grains (city, region, country, continent) while generating interpretable chain-of-thought reasoning.

GeoAgent introduces: 

1. **Geo-similarity reward** combining spatial and semantic similarity to handle the many-to-one mapping between natural language and geographic locations; 
2. **Consistency reward** assessed by a consistency agent to ensure the integrity and consistency of reasoning chains. The model is trained on **GeoSeek**, a novel geolocation dataset with human-annotated CoT and bias-reducing sampling.

We also introduce **GeoSeek**, which is a new geolocation dataset comprising:

- **GeoSeek-CoT** (10k): High-quality chain-of-thought data labeled by geography experts and professional geolocation game players. Each entry includes street-view images, GPS coordinates, three-level location labels (country, city, precise location), and human reasoning processes—standardized into a unified CoT format.
- **GeoSeek-Loc** (20k): Images for RL-based finetuning, sampled via a stratified strategy considering population, land area, and highway mileage to reduce geographic bias.
- **GeoSeek-Val** (3k): Validation benchmark with locatability scores and scene categories (manmade structures, natural landscapes, etc.) for evaluation.



<!-- <div align="center">
<img src="assets/depthanything-AC-video.gif" alt="video" width="100%">
</div> -->

## News

<!-- **2025-07-03:** 🚀  DepthAnything-AC was ranked **#3 Paper of the Day** on [HuggingFace Daily Papers](https://huggingface.co/papers/date/2025-07-03).
- **2025-07-03:** 🔥  The paper of [DepthAnything-AC](https://arxiv.org/abs/2507.01634) is released.  -->
- **2026-02-13:**  🔥 The code of GeoAgent is released.
- **2026-02-13:** 🔥  The GeoAgent model and GeoSeek dataset are released 

## TODO List
- [ ] Instructions for training dataset download and process.
- [ ] Jittor implementation of GeoAgent.
- [ ] Release video demo.

## Model Architecture

![architecture](assets/pipeline.png)

## Installation

### Requirements

- Python>=3.9
- torch==2.6.0
- torchvision==0.21.0
- torchaudio==2.6.0
- ms-swift>=3.8.0
- xformers==0.0.27.post2 
- deepspeed==0.15.0
- cuda==12.4

### Setup
```bash
git clone https://github.com/HVision-NKU/GeoAgent.git
cd GeoAgent

conda create -n GeoAgent python=3.9
conda activate GeoAgent
pip install -r requirements.txt
```

## Usage
### Get GeoAgent Model
Download the pre-trained checkpoints from [Hugging Face](https://huggingface.co/ghost233lism/GeoAgent):
```bash
mkdir checkpoints
cd checkpoints

# (Optional) Using huggingface mirrors
export HF_ENDPOINT=https://hf-mirror.com

# download GeoAgent model from huggingface
huggingface-cli download --resume-download ghost233lism/GeoAgent --local-dir ghost233lism/GeoAgent
```

### Quick Inference

We provide the quick inference scripts for single/batch image input in `infer/`.  Please refer to [infer/README](infer/README.md) for detailed information.

### Training


```bash
bash tools/train_sft.sh 
bash tools/train_grpo.sh
```




## Citation

```bibtex
@article{jin2025geoagent,
  title={GeoAgent: Learning to Geolocate Everywhere with Reinforced Geographic Characteristics},
  author={Jin, Modi and Zhang, Yiming and Sun, Boyuan and Zhang, Dingwen and Cheng, Ming-Ming and Hou, Qibin},
  journal={arXiv preprint arXiv:2602.12617},
  year={2026}
}
```


## License

This code is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) for non-commercial use only.

Please note that any commercial use of this code requires formal permission prior to use.

## Contact

For technical questions, please contact jin_modi[AT]mail.nankai.edu.cn

For commercial licensing, please contact andrewhoux[AT]gmail.com.

## Acknowledgments

We sincerely thank [Yue Zhang](https://tuxun.fun/), [H.M.](https://space.bilibili.com/1655209518?spm_id_from=333.337.0.0), [Haowen He](https://space.bilibili.com/111714204?spm_id_from=333.337.0.0), [Yuke Jun](https://space.bilibili.com/93569847?spm_id_from=333.337.0.0), and other experts in geography, as well as outstanding geolocation game players, for their valuable guidance, prompt design suggestions, and data support throughout the construction of the GeoSeek dataset.

We also thank [Zhixiang Wang](https://tuxun.fun/), [Chilin Chen](https://tuxun.fun/), [Jincheng Shi](https://tuxun.fun/), [Liupeng Zhang](https://tuxun.fun/), [Yuan Gu](https://tuxun.fun/), [Yanghang Shao](https://tuxun.fun/), [Jinhua Zhang](https://tuxun.fun/), [Jiachen Zhu](https://tuxun.fun/), [Gucheng Qiuyue](https://tuxun.fun/), [Qingyang Guo](https://tuxun.fun/), [Jingchen Yang](https://tuxun.fun/), [Weilong Kong](https://tuxun.fun/), [Xinyuan Li](https://tuxun.fun/), and [Mr. Xu](https://tuxun.fun/) (an anonymous volunteer) 
for their outstanding contributions in providing high-quality reasoning process data.

<p align="center">
  <img src="https://api.star-history.com/svg?repos=HVision-NKU/GeoAgent&type=Date" style="width:70%"/>
</p>



