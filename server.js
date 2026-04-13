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

// AI Route using Google Gemini
app.post('/api/analyze', async (req, res) => {
    try {
        console.log("Received body:", JSON.stringify(req.body).substring(0, 200));

        // Check API key
        if (!process.env.GEMINI_API_KEY) {
            throw new Error('GEMINI_API_KEY is not set');
        }

        // Extract the messages from whatever format the frontend sends
        let systemPrompt = '';
        let userMessage = '';

        if (req.body.messages) {
            // Frontend is sending full messages array (current format)
            const msgs = req.body.messages;
            userMessage = msgs.map(m => m.content).join('\n');
            systemPrompt = req.body.system || '';
        } else {
            // Fallback: features format
            const features = req.body.features || req.body;
            const modelType = req.body.modelType || 'Unknown';
            const predicted_class = features.predicted_class ?? features.prediction ?? 'Unknown';
            const proba = features.proba ?? features.confidence ?? 'N/A';
            const importance = features.importance ?? {};
            systemPrompt = "You are SENTINEL, a strategic defense AI. Provide a concise, professional military risk assessment.";
            userMessage = `ML Analysis:\nPredicted Risk: ${predicted_class}\nConfidence: ${proba}\nModel: ${modelType}\nKey Features:\n${Object.entries(importance).map(([k,v]) => `- ${k}: ${v}`).join('\n')}`;
        }

        // Call Gemini API
        const geminiResponse = await fetch(
            `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${process.env.GEMINI_API_KEY}`,
            {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    contents: [
                        {
                            parts: [
                                { text: systemPrompt ? `${systemPrompt}\n\n${userMessage}` : userMessage }
                            ]
                        }
                    ],
                    generationConfig: {
                        maxOutputTokens: 2500,
                        temperature: 0.7
                    }
                })
            }
        );

        const data = await geminiResponse.json();
        console.log("Gemini response status:", geminiResponse.status);

        if (data.error) throw new Error(data.error.message);

        const analysisText = data.candidates?.[0]?.content?.parts?.[0]?.text;
        if (!analysisText) throw new Error('No response from Gemini');

        res.json({ analysis: analysisText });

    } catch (error) {
        console.error("Gemini Error:", error.message);
        res.status(500).json({ error: error.message || "Failed to generate AI report." });
    }
});

// Health check
app.get('/health', (req, res) => res.json({
    status: 'ok',
    system: 'SENTINEL v3',
    apiKeyConfigured: !!process.env.GEMINI_API_KEY,
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
    console.log(`🔑 Gemini Key: ${process.env.GEMINI_API_KEY ? 'SET ✓' : 'MISSING ✗'}`);
});