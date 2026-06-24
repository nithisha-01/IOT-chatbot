#!/bin/bash
# setup.sh — gets you from zero to running in one go (Linux/Mac).
# On Windows, just follow the README steps manually.
set -e

echo "== Chat With Data — setup =="

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env — open it and paste your GROQ_API_KEY before continuing."
  exit 0
fi

echo "1. Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "2. Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "3. Starting PostgreSQL via Docker..."
docker compose up -d
echo "   Waiting 8s for Postgres to be ready..."
sleep 8

echo "4. Loading sample smart-building sensor data..."
python load_sample_data.py

echo "5. Starting the API server (Ctrl+C to stop)..."
cd app
uvicorn main:app --reload --port 8000
