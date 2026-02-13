# Geolocation Inference

**English | [简体中文](README_zh.md)**

Unified inference script for geolocation prediction, supporting single image and batch folder inference.

## Features

- **Single image mode**: Predict location for one image
- **Batch folder mode**: Process all images in a directory recursively
- **Output format**: JSON with Chain-of-Thought reasoning and FinalAnswer (Country; Region; Specific Location)

## Requirements

- Python >= 3.9
- PyTorch, CUDA (for GPU inference)
- [swift](https://github.com/modelscope/swift) (MS-Swift framework)
- tqdm

## Usage

### Single Image Inference

```bash
python infer.py one --image_path /path/to/image.jpg --model_path /path/to/Qwen2.5-VL-model
```

**Arguments:**

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--image_path` | Yes | - | Path to input image |
| `--model_path` | No | YOUR_MODEL_PATH | Qwen2.5-VL model directory |
| `--max_new_tokens` | No | 2048 | Maximum tokens to generate |

### Batch Folder Inference

```bash
python infer.py folder --image_dir /path/to/images --model_path /path/to/Qwen2.5-VL-model
```

**Arguments:**

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--image_dir` | Yes | - | Directory containing images (recursive) |
| `--model_path` | No | YOUR_MODEL_PATH | Qwen2.5-VL model directory |
| `--max_new_tokens` | No | 2048 | Maximum tokens to generate |
| `--output_dir` | No | ./folder_infer_results | Directory to save results |
| `--overwrite` | No | False | Overwrite existing results file |

**Supported image formats:** `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`

## Output

### Single Image Mode

Prints JSON output to stdout with:
- `ChainOfThought`: Reasoning steps (CountryIdentification, RegionalGuess, PreciseLocalization)
- `FinalAnswer`: Format `Country; Region; Specific Location`

### Batch Mode

Saves results to `{output_dir}/results_{timestamp}.json` with structure:

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
    "error": "error message"
  }
]
```

**Resume support:** If the output file already exists and `--overwrite` is not set, completed images are skipped.

## Example

```bash
# Single image
python infer.py one --image_path ./test.jpg --model_path ./checkpoints/Qwen2.5-VL-7B-Instruct

# Batch folder
python infer.py folder --image_dir ./my_photos --model_path ./checkpoints/Qwen2.5-VL-7B-Instruct --output_dir ./results
```
