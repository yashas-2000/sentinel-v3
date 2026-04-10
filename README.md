# SENTINEL v3 — Defense Risk Intelligence System

> AI-powered geopolitical risk analysis, conflict trend detection, military buildup classification, information warfare monitoring, and strategic decision support.

---

## Prerequisites

| Requirement | Minimum Version | Check |
|---|---|---|
| Node.js | 18.x LTS | `node -v` |
| npm | 9.x | `npm -v` |
| Anthropic API Key | — | [console.anthropic.com](https://console.anthropic.com) |

---

## Quick Start (3 steps)

### Step 1 — Clone / download the project

If you downloaded as a ZIP, extract it. If using git:

```bash
git clone <your-repo-url>
cd sentinel
```

### Step 2 — Run the setup script

```bash
bash setup.sh
```

This will:
- Check your Node.js version
- Install all dependencies (`express`, `cors`, `dotenv`, etc.)
- Create a `.env` file from the template

### Step 3 — Add your API key

Open the newly created `.env` file:

```env
ANTHROPIC_API_KEY=sk-ant-your-key-here   ← replace this
PORT=3001
FRONTEND_ORIGIN=http://localhost:3000
RATE_LIMIT_RPM=30
```

Get your key at → [https://console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys)

Then start the server:

```bash
npm start
```

Open your browser at **http://localhost:3001**

---

## Manual Setup (if setup.sh doesn't work)

```bash
# 1. Install dependencies
npm install

# 2. Create .env
cp .env.example .env

# 3. Edit .env and add your API key
#    Open .env in any text editor

# 4. Start server
npm start
```

---

## Project Structure

```
sentinel/
├── server.js          ← Express backend proxy (keeps API key secure)
├── package.json       ← Node.js dependencies
├── .env.example       ← Environment template (copy → .env)
├── .env               ← Your actual config (never commit this!)
├── .gitignore         ← Protects .env from git
├── setup.sh           ← First-time setup script
├── README.md          ← This file
└── public/
    └── index.html     ← SENTINEL frontend (served by Express)
```

---

## How It Works

```
Browser (index.html)
        │
        │  POST /api/analyze
        │  { model, messages, system }
        ▼
Express Server (server.js :3001)
        │
        │  Injects ANTHROPIC_API_KEY header
        │  POST https://api.anthropic.com/v1/messages
        ▼
Anthropic Claude API
        │
        │  Returns analysis JSON
        ▼
Express → Browser
```

**Why a backend proxy?**
Browsers block direct calls to `api.anthropic.com` due to CORS policy. More importantly, calling the API directly from the browser would **expose your API key** in the page source. The backend proxy keeps the key on the server only.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Check server status |
| POST | `/api/analyze` | Proxy to Anthropic API |
| GET | `/*` | Serves the frontend |

### Health check response
```json
{
  "status": "online",
  "system": "SENTINEL v3",
  "model": "claude-sonnet-4-20250514",
  "timestamp": "2026-03-23T10:00:00.000Z",
  "apiKeyConfigured": true
}
```

### Analyze request body
```json
{
  "model": "claude-sonnet-4-20250514",
  "max_tokens": 2500,
  "system": "You are SENTINEL v3...",
  "messages": [
    { "role": "user", "content": "..." }
  ]
}
```

---

## Configuration (.env)

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | *(required)* | Your Anthropic API key |
| `PORT` | `3001` | Backend server port |
| `FRONTEND_ORIGIN` | `http://localhost:3000` | Allowed CORS origin |
| `RATE_LIMIT_RPM` | `30` | Max requests/min per IP |

---

## Development Mode (auto-restart on file changes)

```bash
npm run dev
```

Requires `nodemon` (included in devDependencies).

---

## SENTINEL v3 Modules

| Module | ML Algorithm | Primary Dataset |
|---|---|---|
| Risk Assessment | Random Forest / XGBoost / SVM | SIPRI, World Bank PSI, ACLED |
| Conflict Analysis | XGBoost | ACLED disaggregated events |
| Military Buildup | XGBoost ensemble | SIPRI TIV, OSINT, satellite |
| Information Warfare | TF-IDF + SVM | GDELT, RAND IO corpus |
| Strategic Decision | Claude LLM + cross-module state | All modules combined |

---

## Troubleshooting

**"ANTHROPIC_API_KEY is not set" error on startup**
→ Make sure `.env` exists and contains your real key (not the placeholder).

**"CORS: origin not allowed" in browser console**
→ Add your frontend URL to `FRONTEND_ORIGIN` in `.env`. For `file://` opened HTML, the origin is `null` which is allowed by default.

**"Rate limit exceeded"**
→ Increase `RATE_LIMIT_RPM` in `.env` or wait 1 minute.

**Port already in use**
→ Change `PORT=3001` to another port (e.g. `3002`) in `.env`.

**Node.js version error**
→ Install Node.js 18 LTS from [nodejs.org](https://nodejs.org). Use `nvm` to manage multiple versions.

---

## Security Notes

- ✅ API key stored in `.env`, never sent to the browser
- ✅ CORS restricts which origins can call the backend
- ✅ Rate limiting prevents API key abuse
- ✅ Helmet sets secure HTTP headers
- ✅ `.gitignore` prevents `.env` from being committed
- ⚠️ For production deployment, use environment variables from your host (not a `.env` file)

---

## Disclaimer

SENTINEL v3 is an academic research prototype for the study of ML-based geopolitical risk modeling. All analysis is AI-generated for educational purposes only. It does not constitute real intelligence, military advice, or policy recommendations.
