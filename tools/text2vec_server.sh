#!/bin/bash
# Start the text2vec service with Gunicorn (multi-process mode, supports 8 parallel instances)

# ============ Configuration Parameters ============
# Only need to change this parameter to switch between models!
# Options: "chinese" (Chinese model) or "multilingual" (multilingual model)
MODEL_TYPE="multilingual"
# ==================================

HOST="0.0.0.0"
WORKERS=8  # 8 worker processes, supports 8 concurrent training instances

# Automatically set the port based on the model type
if [ "$MODEL_TYPE" = "multilingual" ]; then
    PORT=5002
else
    PORT=5000
fi

# Check if gunicorn is installed
python3 -c "import gunicorn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Gunicorn is not installed, installing..."
    pip install gunicorn
    if [ $? -ne 0 ]; then
        echo "Failed to install Gunicorn"
        exit 1
    fi
    echo "✅ Gunicorn installed successfully"
fi

# Check if Flask is installed
python3 -c "import flask" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Flask is not installed, installing..."
    pip install flask
fi


# Export model type environment variable (accessible by every worker)
export TEXT2VEC_MODEL_TYPE="$MODEL_TYPE"

# Show model information
if [ "$MODEL_TYPE" = "multilingual" ]; then
    MODEL_DESC="paraphrase-multilingual-MiniLM-L12-v2 (multilingual model)"
else
    MODEL_DESC="text2vec-base-chinese (Chinese model)"
fi

echo "=========================================="
echo "Starting text2vec service (Gunicorn multi-process mode)"
echo "=========================================="
echo "Model Type: $MODEL_TYPE"
echo "Model Description: $MODEL_DESC"
echo "Listening Address: $HOST:$PORT"
echo "Worker Processes: $WORKERS"
echo "Concurrency Support: Can handle $WORKERS requests at the same time"
echo "Timeout: 120 seconds"
echo ""
echo "✅ Configuration complete. Starting service..."
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Switch to script directory (to ensure text2vec_server.py can be found)
cd "$SCRIPT_DIR"

# Start the service
gunicorn text2vec_server:app \
    --workers $WORKERS \
    --threads 4 \
    --bind $HOST:$PORT \
    --timeout 120 \
    --worker-class gthread \
    --access-logfile - \
    --error-logfile - \
    --log-level info

