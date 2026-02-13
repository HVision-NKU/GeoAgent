# 地理位置推理

**[English](README.md) | 简体中文**

地理定位推理脚本，支持单张图片推理和文件夹批量推理。

## 功能

- **单张图片模式**：对单张图片进行地理位置预测
- **批量文件夹模式**：递归处理目录下所有图片
- **输出格式**：JSON 格式，包含思维链推理和最终答案（国家；地区；具体位置）

## 环境要求

- Python >= 3.9
- PyTorch、CUDA（GPU 推理）
- [swift](https://github.com/modelscope/swift)（MS-Swift 框架）
- tqdm

## 使用方法

### 单张图片推理

```bash
python infer.py one --image_path /path/to/image.jpg --model_path /path/to/Qwen2.5-VL-model
```

**参数说明：**

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `--image_path` | 是 | - | 输入图片路径 |
| `--model_path` | 否 | YOUR_MODEL_PATH | Qwen2.5-VL 模型目录 |
| `--max_new_tokens` | 否 | 2048 | 最大生成 token 数 |

### 批量文件夹推理

```bash
python infer.py folder --image_dir /path/to/images --model_path /path/to/Qwen2.5-VL-model
```

**参数说明：**

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `--image_dir` | 是 | - | 输入图片文件夹（递归扫描） |
| `--model_path` | 否 | YOUR_MODEL_PATH | Qwen2.5-VL 模型目录 |
| `--max_new_tokens` | 否 | 2048 | 最大生成 token 数 |
| `--output_dir` | 否 | ./folder_infer_results | 结果保存目录 |
| `--overwrite` | 否 | False | 覆盖已存在的结果文件 |

**支持的图片格式：** `.jpg`、`.jpeg`、`.png`、`.bmp`、`.tif`、`.tiff`

## 输出说明

### 单张图片模式

输出到标准输出的 JSON 包含：
- `ChainOfThought`：推理过程（国家识别、区域猜测、精确定位）
- `FinalAnswer`：格式为 `国家; 地区; 具体位置`

### 批量模式

结果保存至 `{output_dir}/results_{timestamp}.json`，结构如下：

```json
[
  {
    "image": "/path/to/image.jpg",
    "raw_response": "{...}",
    "status": "success"
  },
  {
    "image": "/path/to/another.jpg",
    "raw_response": "",
    "status": "error",
    "error": "错误信息"
  }
]
```

**断点续跑：** 若输出文件已存在且未使用 `--overwrite`，则跳过已完成图片。


## 示例

```bash
# 单张图片
python infer.py one --image_path ./test.jpg --model_path ./checkpoints/Qwen2.5-VL-7B-Instruct

# 批量文件夹
python infer.py folder --image_dir ./my_photos --model_path ./checkpoints/Qwen2.5-VL-7B-Instruct --output_dir ./results
```
