#!/bin/bash
# ══════════════════════════════════════════════════════════════
#  SENTINEL v3 — Python ML Backend Setup
#  Run from the project root: bash setup_ml.sh
# ══════════════════════════════════════════════════════════════

set -e
GREEN='\033[0;32m'; AMBER='\033[0;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║    SENTINEL v3 — Python ML Setup              ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════╝${NC}"
echo ""

# 1. Check Python
echo -e "${AMBER}[1/4]${NC} Checking Python..."
if ! command -v python3 &>/dev/null && ! command -v python &>/dev/null; then
  echo -e "${RED}✗ Python not found. Install Python 3.9+ from https://python.org${NC}"
  exit 1
fi
PY=$(command -v python3 || command -v python)
echo -e "${GREEN}✓ Python found: $($PY --version)${NC}"

# 2. Create virtual environment
echo -e "${AMBER}[2/4]${NC} Creating virtual environment..."
cd ml_backend
if [ ! -d "venv" ]; then
  $PY -m venv venv
  echo -e "${GREEN}✓ Virtual environment created${NC}"
else
  echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi

# Activate
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null

# 3. Install dependencies
echo -e "${AMBER}[3/4]${NC} Installing Python packages..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo -e "${GREEN}✓ Packages installed: flask, scikit-learn, xgboost, numpy, pandas${NC}"

# 4. Train models
echo -e "${AMBER}[4/4]${NC} Downloading real datasets & training ML models (~2-5 min)..."
python fetch_and_train.py
echo -e "${GREEN}✓ All models trained and saved to ml_backend/models/${NC}"

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║     Python ML Backend Ready!                  ║${NC}"
echo -e "${GREEN}╠══════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║                                               ║${NC}"
echo -e "${GREEN}║  Start ML API:  ${CYAN}python app.py${GREEN}               ║${NC}"
echo -e "${GREEN}║  (from inside ml_backend/ folder)             ║${NC}"
echo -e "${GREEN}║                                               ║${NC}"
echo -e "${GREEN}║  Then in a SECOND terminal:                   ║${NC}"
echo -e "${GREEN}║  Start Node:    ${CYAN}npm start${GREEN}  (project root)    ║${NC}"
echo -e "${GREEN}║                                               ║${NC}"
echo -e "${GREEN}║  Open browser:  ${CYAN}http://localhost:3001${GREEN}        ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════╝${NC}"
echo ""
