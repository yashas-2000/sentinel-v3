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

// AI Route using Groq
app.post('/api/analyze', async (req, res) => {
    try {
        console.log("Received body:", JSON.stringify(req.body).substring(0, 200));

        // Check API key
        if (!process.env.GROQ_API_KEY) {
            throw new Error('GROQ_API_KEY is not set');
        }

        // Extract messages from whatever format the frontend sends
        let messages = [];

        if (req.body.messages) {
            // Frontend sends full messages array
            const systemPrompt = req.body.system || '';
            if (systemPrompt) {
                messages.push({ role: 'system', content: systemPrompt });
            }
            messages = messages.concat(req.body.messages);
        } else {
            // Fallback: features format
            const features = req.body.features || req.body;
            const modelType = req.body.modelType || 'Unknown';
            const predicted_class = features.predicted_class ?? features.prediction ?? 'Unknown';
            const proba = features.proba ?? features.confidence ?? 'N/A';
            const importance = features.importance ?? {};
            messages = [
                {
                    role: 'system',
                    content: 'You are SENTINEL, a strategic defense AI. Provide a concise, professional military risk assessment.'
                },
                {
                    role: 'user',
                    content: `ML Analysis:\nPredicted Risk: ${predicted_class}\nConfidence: ${proba}\nModel: ${modelType}\nKey Features:\n${Object.entries(importance).map(([k,v]) => `- ${k}: ${v}`).join('\n')}`
                }
            ];
        }

        // Call Groq API (OpenAI-compatible format)
        const groqResponse = await fetch('https://api.groq.com/openai/v1/chat/completions', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${process.env.GROQ_API_KEY}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                model: 'llama-3.3-70b-versatile',
                messages: messages,
                max_tokens: 2500,
                temperature: 0.7
            })
        });

        const data = await groqResponse.json();
        console.log("Groq response status:", groqResponse.status);

        if (data.error) throw new Error(data.error.message);

        const analysisText = data.choices?.[0]?.message?.content;
        if (!analysisText) throw new Error('No response from Groq');

        res.json({ analysis: analysisText });

    } catch (error) {
        console.error("Groq Error:", error.message);
        res.status(500).json({ error: error.message || "Failed to generate AI report." });
    }
});

// Health check
app.get('/health', (req, res) => res.json({
    status: 'ok',
    system: 'SENTINEL v3',
    apiKeyConfigured: !!process.env.GROQ_API_KEY,
    mlUrl: process.env.ML_API_URL || 'localhost:5001'
}));

// Serve frontend
app.get('*', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Start server
app.listen(PORT, () => {
    console.log(`🚀 SENTINEL Server running at http://localhost:${PORT}`);
    console.log(`🔗 ML Backend: ${process.env.ML_API_URL || 'http://localhost:5001'}`);
    console.log(`🔑 Groq Key: ${process.env.GROQ_API_KEY ? 'SET ✓' : 'MISSING ✗'}`);
});