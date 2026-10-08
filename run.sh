#!/bin/bash
# Lito Runner for Linux/Mac

echo "============================================"
echo "  Lito - Artificial Intelligence"
echo "  Built From Scratch - No Pretrained Models"
echo "============================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed."
    echo "Install with: sudo apt install python3 python3-pip"
    exit 1
fi

# Install dependencies
echo "Installing dependencies..."
pip3 install numpy --quiet 2>/dev/null

# Check if model exists
if [ -f "checkpoints/best_model.npz" ]; then
    echo "Trained model found!"
    echo ""
    read -p "Start chatting? (Y/N, or T to retrain): " CHOICE
    case $CHOICE in
        [Tt]) ;;
        [Nn]) exit 0 ;;
        *)
            python3 main.py --dir checkpoints
            exit 0
            ;;
    esac
fi

# Train
echo ""
echo "Training AI Model..."
echo ""
python3 train.py --epochs 30 --batch_size 4 --seq_len 64

echo ""
echo "Starting Lito Chat..."
python3 main.py --dir checkpoints