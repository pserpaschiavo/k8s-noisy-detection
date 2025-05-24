#!/bin/bash
# Quick setup script for k8s-noisy-detection

echo "🚀 Setting up k8s-noisy-detection..."

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Install dependencies
echo "📦 Installing dependencies..."
pip3 install -r requirements.txt

# Install in development mode
echo "🔧 Installing package in development mode..."
pip3 install -e .

echo "✅ Setup complete!"
echo ""
echo "🎯 Quick Start:"
echo "  python3 -m src.main --data-dir demo-data/demo-experiment-1-round --output-dir output"
echo ""
echo "📚 For more options, run:"
echo "  python3 -m src.main --help"
