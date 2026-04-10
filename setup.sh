#!/bin/bash
# ══════════════════════════════════════════════════════════════
#  SENTINEL v3 — First-Time Setup Script
#  Run this once: bash setup.sh
# ══════════════════════════════════════════════════════════════

set -e  # Exit on any error

GREEN='\033[0;32m'
AMBER='\033[0;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║      SENTINEL v3  —  Setup Initializing       ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════╝${NC}"
echo ""

# ── 1. Check Node.js ────────────────────────────────────────
echo -e "${AMBER}[1/5]${NC} Checking Node.js version..."
if ! command -v node &> /dev/null; then
  echo -e "${RED}✗ Node.js not found.${NC}"
  echo "  Install from https://nodejs.org (LTS version recommended, ≥18)"
  exit 1
fi

NODE_VERSION=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 18 ]; then
  echo -e "${RED}✗ Node.js version $(node -v) is too old. Need ≥18.${NC}"
  exit 1
fi
echo -e "${GREEN}✓ Node.js $(node -v) found${NC}"

# ── 2. Check npm ─────────────────────────────────────────────
echo -e "${AMBER}[2/5]${NC} Checking npm..."
if ! command -v npm &> /dev/null; then
  echo -e "${RED}✗ npm not found. It should come with Node.js.${NC}"
  exit 1
fi
echo -e "${GREEN}✓ npm $(npm -v) found${NC}"

# ── 3. Install dependencies ──────────────────────────────────
echo -e "${AMBER}[3/5]${NC} Installing Node.js dependencies..."
npm install --silent
echo -e "${GREEN}✓ Dependencies installed${NC}"

# ── 4. Create .env from template ─────────────────────────────
echo -e "${AMBER}[4/5]${NC} Setting up environment file..."

if [ -f ".env" ]; then
  echo -e "${GREEN}✓ .env file already exists — skipping${NC}"
else
  cp .env.example .env
  echo -e "${GREEN}✓ .env file created from template${NC}"
  echo ""
  echo -e "${RED}  ⚠  ACTION REQUIRED:${NC}"
  echo "     Open .env in a text editor and replace the placeholder:"
  echo "     ANTHROPIC_API_KEY=sk-ant-your-key-here"
  echo "     with your real API key from https://console.anthropic.com"
  echo ""
fi

# ── 5. Verify .env has a real key ────────────────────────────
echo -e "${AMBER}[5/5]${NC} Validating API key in .env..."

if grep -q "sk-ant-your-key-here" .env; then
  echo -e "${RED}✗ API key is still the placeholder value.${NC}"
  echo "  Edit .env and set a real ANTHROPIC_API_KEY before starting."
  echo ""
  echo -e "${CYAN}  Once done, start the server with:${NC}  npm start"
  echo ""
  exit 0
fi

if ! grep -q "ANTHROPIC_API_KEY=sk-ant" .env; then
  echo -e "${RED}✗ Could not find ANTHROPIC_API_KEY in .env${NC}"
  echo "  Make sure your .env contains: ANTHROPIC_API_KEY=sk-ant-..."
  exit 1
fi

echo -e "${GREEN}✓ API key found in .env${NC}"

# ── ALL DONE ─────────────────────────────────────────────────
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║          Setup Complete — Ready to Launch!    ║${NC}"
echo -e "${GREEN}╠══════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║                                               ║${NC}"
echo -e "${GREEN}║  Start server:   ${CYAN}npm start${GREEN}                   ║${NC}"
echo -e "${GREEN}║  Dev mode:       ${CYAN}npm run dev${GREEN}                 ║${NC}"
echo -e "${GREEN}║  Then open:      ${CYAN}http://localhost:3001${GREEN}        ║${NC}"
echo -e "${GREEN}║                                               ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════╝${NC}"
echo ""
