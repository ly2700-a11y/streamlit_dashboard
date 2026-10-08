#!/bin/bash
# Launch Streamlit app with virtual environment activation

cd "$(dirname "$0")"

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "❌ Virtual environment not found at .venv/"
    echo "Please create one with: python3 -m venv .venv"
    exit 1
fi

# Launch Streamlit app
streamlit run src/app/main.py
