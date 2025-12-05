#!/bin/bash
# BERT Training Monitor - Run this in a separate terminal to watch progress

echo "=========================================="
echo "BERT Training Monitor"
echo "=========================================="
echo ""

# Check if training is running
echo "=== Process Status ==="
PROCESS=$(ps aux | grep "train_bert" | grep -v grep)
if [ -z "$PROCESS" ]; then
    echo "❌ Training process NOT running"
else
    echo "✅ Training process IS running"
    echo "$PROCESS" | awk '{print "  PID: " $2 ", CPU: " $3 "%, Memory: " $4 "%"}'
fi
echo ""

# Check model download
echo "=== Model Download Status ==="
CACHE_DIR="$HOME/.cache/huggingface/hub/models--bert-base-uncased"
if [ -d "$CACHE_DIR" ]; then
    MODEL_FILE=$(find "$CACHE_DIR" -name "*.safetensors" -o -name "pytorch_model.bin" 2>/dev/null | head -1)
    if [ -n "$MODEL_FILE" ] && [ -f "$MODEL_FILE" ]; then
        SIZE=$(du -h "$MODEL_FILE" | cut -f1)
        echo "✅ Model weights downloaded: $SIZE"
    else
        echo "⏳ Model weights still downloading..."
    fi
    
    TOKENIZER_FILES=$(find "$CACHE_DIR" -name "vocab.txt" -o -name "tokenizer.json" 2>/dev/null | wc -l | tr -d ' ')
    echo "✅ Tokenizer files: $TOKENIZER_FILES files"
else
    echo "⏳ Cache directory not found (download starting...)"
fi
echo ""

# Check output files
echo "=== Training Output Files ==="
OUTPUT_DIR="runs/bert_title"
if [ -d "$OUTPUT_DIR" ]; then
    FILE_COUNT=$(ls -1 "$OUTPUT_DIR" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$FILE_COUNT" -gt 0 ]; then
        echo "✅ Output directory exists with $FILE_COUNT file(s):"
        ls -lht "$OUTPUT_DIR" | head -5 | tail -4
    else
        echo "⏳ Output directory exists but empty (training not started yet)"
    fi
else
    echo "⏳ Output directory not created yet"
fi
echo ""

# Check for log files
echo "=== Check for Logs ==="
if [ -f "runs/bert_title/trainer_state.json" ]; then
    echo "✅ Training state file found - training has started!"
    # Try to extract progress if possible
    if command -v python3 &> /dev/null; then
        python3 -c "
import json
try:
    with open('runs/bert_title/trainer_state.json') as f:
        state = json.load(f)
        if 'log_history' in state and len(state['log_history']) > 0:
            latest = state['log_history'][-1]
            print(f\"  Latest log entry: epoch {latest.get('epoch', 'N/A')}, step {latest.get('step', 'N/A')}\")
            if 'loss' in latest:
                print(f\"  Current loss: {latest['loss']:.4f}\")
except:
    pass
" 2>/dev/null
    fi
else
    echo "⏳ No training state file yet"
fi
echo ""

echo "=========================================="
echo "Run this script again to refresh status"
echo "Or use: watch -n 5 ./monitor_training.sh"
echo "=========================================="

