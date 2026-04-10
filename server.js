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

// ML Backend Proxy (Flask on :5001)
app.post('/api/predict', async (req, res) => {
    try {
        const response = await fetch('http://localhost:5001/predict', {
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
        const { features, modelType } = req.body;

        const response = await fetch("https://openrouter.ai/api/v1/chat/completions", {
            method: "POST",
            headers: {
                "Authorization": `Bearer ${process.env.OPENROUTER_API_KEY}`,
                "HTTP-Referer": "http://localhost:3001",
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
                        "content": `ML Analysis:\nPredicted Risk: ${features.predicted_class}\nConfidence: ${features.proba}\nModel: ${modelType}\nKey Features:\n${Object.entries(features.importance || {}).map(([k,v]) => `- ${k}: ${v}`).join('\n')}`
                    }
                ]
            })
        });

        const data = await response.json();
        if (data.error) throw new Error(data.error.message);

        res.json({ 
            analysis: data.choices[0].message.content 
        });

    } catch (error) {
        console.error("OpenRouter Error:", error);
        res.status(500).json({ error: "Failed to generate AI report." });
    }
});

// Health check
app.get('/health', (req, res) => res.json({ status: 'ok' }));

// Serve frontend
app.get('*', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Start the server
app.listen(PORT, () => {
    console.log(`🚀 SENTINEL Server running at http://localhost:${PORT}`);
    console.log(`🔗 ML Backend Proxy: http://localhost:5001`);
    console.log(`🔗 OpenRouter AI Ready`);
});
