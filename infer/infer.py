#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified geolocation inference script
Supports single image inference and batch folder inference
"""
import os
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

from swift.llm import (
    PtEngine, RequestConfig,
    get_model_tokenizer, get_template, InferRequest
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.JPG', '.JPEG', '.PNG'}


class SingleImageInfer:
    """Geolocation inference for a single image"""

    def __init__(self,
                 model_path: str,
                 max_new_tokens: int = 2048):
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.engine = None
        self.template = None
        self._load_model()

    def _load_model(self):
        logger.info("Loading model & tokenizer...")
        model, tokenizer = get_model_tokenizer(self.model_path, model_type='qwen2_5_vl')

        template_type = model.model_meta.template
        self.template = get_template(template_type, tokenizer,
                                     default_system=self._system_prompt())
        self.engine = PtEngine.from_model_template(model, self.template, max_batch_size=1)
        logger.info("Model ready.")

    @staticmethod
    def _system_prompt():
        return """You are an expert with rich experience in the field of geolocation, skilled at accurately locating the geographic location of images through various clues in the images, such as traffic signs, architectural styles, natural landscapes, etc. At the same time, you are also a mentor in building the chain of thought, able to organize complex ideas into clear and standardized patterns. You possess knowledge in multiple disciplines such as geography, cartography, transportation, and architecture, and are able to identify the characteristics of different countries, regions, and locations. At the same time, you have the ability to analyze logic and construct a chain of thought. Task: Output the thought chain and final answer based on the four images input by the user. The thought chain includes:
        Country Identification/Regional Guess/Precise Localization.
        Possible clues include: National clues: (Example: traffic sign shape/color, language and text, driving direction, architectural style, vegetation and climate characteristics, etc.)
        Regional clues: (logo/enterprise, topography, vegetation type, regional traffic signs, dialect/spelling, license plate style, area code/postal code, infrastructure features, etc.)
        Accurate positioning: (road sign text, street name, house number, landmark building, river and lake water system, place attributes such as park/city/commercial district, shop name and storefront, etc.)
        Do not output objects that do not exist in the image.
        Output strictly in JSON format:
        {
        "ChainOfThought": {
            "CountryIdentification": {
            "Clues": [],
            "Reasoning": "",
            "Conclusion": "",
            "Uncertainty": ""
            },
            "RegionalGuess": {
            "Clues": [],
            "Reasoning": "",
            "Conclusion": "",
            "Uncertainty": ""
            },
            "PreciseLocalization": {
            "Clues": [],
            "Reasoning": "",
            "Conclusion": "",
            "Uncertainty": ""
            }
        },
        "FinalAnswer": "Country; Region; Specific Location"
        }"""

    def infer(self, image_path: str) -> str:
        if not os.path.isfile(image_path):
            raise FileNotFoundError(image_path)

        request = InferRequest(
            messages=[{'role': 'user',
                       'content': '<image>Based on the image, tell me the specific location and your thinking process'}],
            images=[image_path]
        )
        cfg = RequestConfig(max_tokens=self.max_new_tokens, temperature=0)
        resp = self.engine.infer([request], cfg)[0]
        print(resp.choices[0].message.content)
        return resp.choices[0].message.content


def collect_images(folder: str):
    """Return a list of absolute paths of all images in the folder"""
    return [str(p) for p in Path(folder).rglob('*') if p.suffix in IMG_EXTS]


def infer_one(args):
    """Single image inference"""
    predictor = SingleImageInfer(
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens
    )
    result = predictor.infer(args.image_path)
    print('\n========== Model Output ==========')
    try:
        print(json.dumps(json.loads(result), ensure_ascii=False, indent=2))
    except json.JSONDecodeError:
        print(result)
    print('===============================')


def infer_folder(args):
    """Batch image inference from a folder"""
    predictor = SingleImageInfer(
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens
    )

    img_list = collect_images(args.image_dir)
    if not img_list:
        logger.error(f'No images found in {args.image_dir}')
        return
    logger.info(f'Found {len(img_list)} images')

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_file = os.path.join(args.output_dir, f'results_{timestamp}.json')

    if os.path.exists(out_file) and not args.overwrite:
        logger.info(f'Result file already exists, appending/skipping: {out_file}')
        with open(out_file, 'r', encoding='utf-8') as f:
            finished = {item['image'] for item in json.load(f)}
    else:
        finished = set()

    results = []
    pbar = tqdm(img_list, desc='Inferencing')
    for img_path in pbar:
        if img_path in finished:
            continue
        try:
            resp = predictor.infer(img_path)
            results.append({
                "image": img_path,
                "raw_response": resp,
                "status": "success"
            })
        except Exception as e:
            logger.error(f'Failed on {img_path}: {e}')
            results.append({
                "image": img_path,
                "raw_response": "",
                "status": "error",
                "error": str(e)
            })
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f'All completed. Results saved to: {out_file}')


def main():
    parser = argparse.ArgumentParser(
        description='Geolocation inference - Supports single image or batch folder inference'
    )
    subparsers = parser.add_subparsers(dest='mode', help='Inference mode')

    # Single image mode
    p_one = subparsers.add_parser('one', help='Single image inference')
    p_one.add_argument('--image_path', required=True, help='Input image path')
    p_one.add_argument('--model_path', default='YOUR_MODEL_PATH', help='Qwen2.5-VL model directory')
    p_one.add_argument('--max_new_tokens', type=int, default=2048)
    p_one.set_defaults(func=infer_one)

    # Folder batch mode
    p_folder = subparsers.add_parser('folder', help='Batch folder inference')
    p_folder.add_argument('--image_dir', required=True, help='Input folder containing images to be inferred')
    p_folder.add_argument('--model_path', default='YOUR_MODEL_PATH')
    p_folder.add_argument('--max_new_tokens', type=int, default=2048)
    p_folder.add_argument('--output_dir', default='./folder_infer_results', help='Directory to save results')
    p_folder.add_argument('--overwrite', action='store_true', help='Overwrite existing results file')
    p_folder.set_defaults(func=infer_folder)

    args = parser.parse_args()
    if args.mode is None:
        parser.print_help()
        return
    args.func(args)


if __name__ == '__main__':
    main()
