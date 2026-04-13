require('dotenv').config();
const express = require('express');
const cors = require('cors');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// ML Backend Proxy
app.post('/api/predict', async (req, res) => {
    try {
        const ML_URL = process.env.ML_API_URL || 'http://localhost:5001';
        const response = await fetch(`${ML_URL}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(req.body)
        });
        const data = await response.json();
        res.json(data);
    } catch (error) {
        console.error('ML Backend Error:', error.message);
        res.status(500).json({ error: 'ML service unavailable' });
    }
});

// The AI Route for OpenRouter
app.post('/api/analyze', async (req, res) => {
    try {
        // Log exactly what we receive to debug
        console.log("Received body:", JSON.stringify(req.body));

        // Handle both formats: { features, modelType } or flat object
        let features = req.body.features || req.body;
        let modelType = req.body.modelType || 'Unknown';

        // Validate features exists
        if (!features) {
            throw new Error('No features data received');
        }

        // Safe fallbacks for all fields
        const predicted_class = features.predicted_class ?? features.prediction ?? 'Unknown';
        const proba = features.proba ?? features.confidence ?? features.probability ?? 'N/A';
        const importance = features.importance ?? features.feature_importance ?? {};

        // Check API key is set
        if (!process.env.OPENROUTER_API_KEY) {
            throw new Error('OPENROUTER_API_KEY is not set');
        }

        const siteUrl = process.env.RENDER_EXTERNAL_URL || 'https://sentinel-v3-l5ve.onrender.com';

        const response = await fetch("https://openrouter.ai/api/v1/chat/completions", {
            method: "POST",
            headers: {
                "Authorization": `Bearer ${process.env.OPENROUTER_API_KEY}`,
                "HTTP-Referer": siteUrl,
                "X-Title": "SENTINEL",
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                "model": "anthropic/claude-3.5-sonnet:free",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are SENTINEL, a strategic defense AI. Provide a concise, professional military risk assessment based on the provided ML predictions and features. Format as executive summary."
                    },
                    {
                        "role": "user",
                        "content": `ML Analysis:\nPredicted Risk: ${predicted_class}\nConfidence: ${proba}\nModel: ${modelType}\nKey Features:\n${Object.entries(importance).map(([k,v]) => `- ${k}: ${v}`).join('\n')}`
                    }
                ]
            })
        });

        const data = await response.json();

        // Log full OpenRouter response for debugging
        console.log("OpenRouter response:", JSON.stringify(data));

        if (data.error) throw new Error(data.error.message);

        res.json({
            analysis: data.choices[0].message.content
        });

    } catch (error) {
        console.error("OpenRouter Error:", error.message);
        res.status(500).json({ error: error.message || "Failed to generate AI report." });
    }
});

// Health check
app.get('/health', (req, res) => res.json({
    status: 'ok',
    system: 'SENTINEL v3',
    apiKeyConfigured: !!process.env.OPENROUTER_API_KEY,
    mlUrl: process.env.ML_API_URL || 'localhost:5001'
}));

// Serve frontend
app.get('*', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Start the server
app.listen(PORT, () => {
    console.log(`🚀 SENTINEL Server running at http://localhost:${PORT}`);
    console.log(`🔗 ML Backend: ${process.env.ML_API_URL || 'http://localhost:5001'}`);
    console.log(`🔑 OpenRouter Key: ${process.env.OPENROUTER_API_KEY ? 'SET ✓' : 'MISSING ✗'}`);
});
