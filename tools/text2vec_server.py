#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
text2vec model service.
Runs on CPU, provides HTTP interface for text encoding.
Supports two models: chinese and multilingual.
"""

from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import argparse
import os

app = Flask(__name__)

model = None
model_path = None
model_name = None

MODEL_CONFIGS = {
    'chinese': {
        'path': 'YOUR_PATH/text2vec-base-chinese',
        'port': 5000,
        'description': 'text2vec-base-chinese (Chinese optimized)'
    },
    'multilingual': {
        'path': 'YOUR_PATH/paraphrase-multilingual-MiniLM-L12-v2',
        'port': 5001,
        'description': 'paraphrase-multilingual-MiniLM-L12-v2 (multilingual)'
    }
}

def load_model(path):
    """Load the model onto CPU."""
    global model
    import os
    worker_id = os.getpid()
    try:
        print(f"[Worker {worker_id}] Loading model: {path}")
        print(f"[Worker {worker_id}] Device: CPU")
        model = SentenceTransformer(path, device='cpu')
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        print(f"[Worker {worker_id}] ✓ Model loaded successfully")
        return True
    except Exception as e:
        print(f"[Worker {worker_id}] ✗ Model load failed: {e}")
        import traceback
        traceback.print_exc()
        return False

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    if model is None:
        return jsonify({
            'status': 'error',
            'message': 'Model not loaded'
        }), 500
    
    return jsonify({
        'status': 'ok',
        'model_name': model_name,
        'model_path': model_path,
        'device': 'cpu'
    })

@app.route('/encode', methods=['POST'])
def encode():
    """Text encoding endpoint."""
    if model is None:
        return jsonify({
            'status': 'error',
            'message': 'Model not loaded'
        }), 500
    
    try:
        data = request.json
        
        if 'text' in data:
            texts = [data['text']] if isinstance(data['text'], str) else data['text']
        elif 'texts' in data:
            texts = data['texts']
        else:
            return jsonify({
                'status': 'error',
                'message': 'Missing "text" or "texts" field'
            }), 400
        
        if not texts:
            return jsonify({
                'status': 'error',
                'message': 'Empty text list'
            }), 400
        
        with torch.no_grad():
            embeddings = model.encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=True,
                device='cpu'
            )
        
        embeddings_list = embeddings.tolist()
        
        return jsonify({
            'status': 'success',
            'embeddings': embeddings_list,
            'shape': list(embeddings.shape)
        })
        
    except Exception as e:
        import traceback
        error_msg = f"Encoding error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/similarity', methods=['POST'])
def similarity():
    """Compute cosine similarity between two texts."""
    if model is None:
        return jsonify({
            'status': 'error',
            'message': 'Model not loaded'
        }), 500
    
    try:
        data = request.json
        
        if 'text1' not in data or 'text2' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Missing "text1" or "text2" field'
            }), 400
        
        text1 = data['text1']
        text2 = data['text2']
        
        with torch.no_grad():
            embeddings = model.encode(
                [text1, text2],
                show_progress_bar=False,
                convert_to_numpy=True,
                device='cpu'
            )
        
        vec1 = embeddings[0]
        vec2 = embeddings[1]
        
        cosine_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        cosine_sim = float(max(0.0, min(1.0, cosine_sim)))
        
        return jsonify({
            'status': 'success',
            'similarity': cosine_sim
        })
        
    except Exception as e:
        import traceback
        error_msg = f"Similarity error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/batch_similarity', methods=['POST'])
def batch_similarity():
    """Batch compute cosine similarities of text pairs."""
    import datetime
    import threading
    pid = os.getpid()
    tid = threading.get_ident()
    current_time = datetime.datetime.now()
    print(f"[Worker PID={pid} | Thread TID={tid}] Processing started at: {current_time}")

    if model is None:
        return jsonify({
            'status': 'error',
            'message': 'Model not loaded'
        }), 500
    
    try:
        data = request.json
        
        if 'pairs' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Missing "pairs" field (list of [text1, text2])'
            }), 400
        
        pairs = data['pairs']
        if not isinstance(pairs, list):
            return jsonify({
                'status': 'error',
                'message': '"pairs" must be a list'
            }), 400
        
        similarities = []
        
        for pair in pairs:
            if len(pair) != 2:
                similarities.append(0.0)
                continue
            
            text1, text2 = pair
            
            with torch.no_grad():
                embeddings = model.encode(
                    [text1, text2],
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    device='cpu'
                )
            
            vec1 = embeddings[0]
            vec2 = embeddings[1]
            
            cosine_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            cosine_sim = float(max(0.0, min(1.0, cosine_sim)))
            
            similarities.append(cosine_sim)

        current_time = datetime.datetime.now()
        print(f"[Worker PID={pid} | Thread TID={tid}] Processing ended at: {current_time}")
        return jsonify({
            'status': 'success',
            'similarities': similarities
        })
        
    except Exception as e:
        import traceback
        error_msg = f"Batch similarity error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

def init_worker():
    """Gunicorn worker initialization function."""
    global model_path, model_name
    model_type = os.getenv('TEXT2VEC_MODEL_TYPE', 'chinese')
    if model_type in MODEL_CONFIGS:
        model_name = model_type
        model_path = MODEL_CONFIGS[model_type]['path']
        print(f"[Worker {os.getpid()}] Using predefined model: {MODEL_CONFIGS[model_type]['description']}")
    else:
        model_path = os.getenv('TEXT2VEC_MODEL_PATH', MODEL_CONFIGS['chinese']['path'])
        model_name = 'custom'
        print(f"[Worker {os.getpid()}] Using custom model path: {model_path}")
    if not load_model(model_path):
        import sys
        print(f"[Worker {os.getpid()}] Model load failed, exiting")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description='text2vec Model Server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python text2vec_server.py --model chinese
  python text2vec_server.py --model multilingual
  python text2vec_server.py --model_path /path/to/your/model
        """
    )
    parser.add_argument('--model', type=str, choices=['chinese', 'multilingual'],
                       default='chinese',
                       help='Select predefined model: chinese or multilingual')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Custom model path (overrides --model if specified)')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host to bind')
    parser.add_argument('--port', type=int, default=None,
                       help='Port to bind (if not set, will use default for model)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    args = parser.parse_args()
    
    global model_path, model_name
    
    if args.model_path:
        model_path = args.model_path
        model_name = 'custom'
        port = args.port if args.port else 5000
        description = f"Custom model: {model_path}"
    else:
        model_name = args.model
        config = MODEL_CONFIGS[model_name]
        model_path = config['path']
        port = args.port if args.port else config['port']
        description = config['description']
    
    print(f"\nLoading model: {description}")
    if not load_model(model_path):
        print("Model load failed, exiting")
        return

    print(f"\nStarting text2vec service:")
    print(f"  Model type: {model_name}")
    print(f"  Model path: {model_path}")
    print(f"  Address: http://{args.host}:{port}")
    print(f"  Device: CPU")
    print(f"\nAvailable endpoints:")
    print(f"  GET  /health              - Health check")
    print(f"  POST /encode              - Text encoding")
    print(f"  POST /similarity          - Compute similarity between two texts")
    print(f"  POST /batch_similarity    - Batch compute similarity")
    print()
    
    app.run(host=args.host, port=port, debug=args.debug)

if os.getenv('TEXT2VEC_MODEL_TYPE') or os.getenv('TEXT2VEC_MODEL_PATH'):
    init_worker()

if __name__ == '__main__':
    main()
