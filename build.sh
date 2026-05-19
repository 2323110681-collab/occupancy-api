#!/bin/bash
echo "🚀 Iniciando build..."
python --version
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
echo "✅ Build completado"
