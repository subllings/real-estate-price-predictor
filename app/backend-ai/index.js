require('dotenv').config();
const express = require('express');
const axios = require('axios');
const bodyParser = require('body-parser');
const cors = require('cors');
const swaggerUi = require('swagger-ui-express');
const swaggerSpecs = require('./swagger');

const app = express();
const port = process.env.PORT || 5050;

const corsOptions = {
  origin: [
    "https://realestate-ui.azurewebsites.net", // ton frontend React en ligne
    "http://localhost:3000" // pour tests en local
  ],
  methods: ["GET", "POST", "OPTIONS"],
  allowedHeaders: ["Content-Type", "Authorization"],
  credentials: true
};

app.use(cors(corsOptions));
app.use(bodyParser.json());

// === Swagger ===
app.use('/docs', swaggerUi.serve, swaggerUi.setup(swaggerSpecs));

/**
 * @openapi
 * /chat:
 *   post:
 *     summary: Get a response from the Azure OpenAI agent
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             properties:
 *               messages:
 *                 type: array
 *                 items:
 *                   type: object
 *                   properties:
 *                     role:
 *                       type: string
 *                     content:
 *                       type: string
 *     responses:
 *       200:
 *         description: AI response
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 response:
 *                   type: string
 *       400:
 *         description: Missing messages field
 *       500:
 *         description: Failed to get response from Azure OpenAI
 */
app.post('/chat', async (req, res) => {
  const { messages } = req.body;

  if (!messages) {
    return res.status(400).json({ error: 'Missing messages field' });
  }

  try {
    const endpoint = process.env.AZURE_OPENAI_ENDPOINT;
    const fullEndpoint = endpoint.endsWith('/') ? endpoint : endpoint + '/';
    const url = `${fullEndpoint}openai/deployments/${process.env.AZURE_OPENAI_DEPLOYMENT_NAME}/chat/completions?api-version=${process.env.AZURE_OPENAI_API_VERSION}`;

    const response = await axios.post(
      url,
      { messages, temperature: 0.7 },
      {
        headers: {
          'Content-Type': 'application/json',
          'api-key': process.env.AZURE_OPENAI_API_KEY
        }
      }
    );

    const aiMessage = response.data.choices?.[0]?.message?.content;
    res.status(200).json({ response: aiMessage });
  } catch (error) {
    console.error('Azure OpenAI API error:', error.response?.data || error.message);
    res.status(500).json({ error: 'Failed to get response from Azure OpenAI.' });
  }
});

/**
 * @openapi
 * /comment:
 *   post:
 *     summary: Generate comments based on predictions
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             properties:
 *               predictionAll:
 *                 type: number
 *               predictionTop:
 *                 type: number
 *     responses:
 *       200:
 *         description: Comments generated
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 comments:
 *                   type: array
 *                   items:
 *                     type: string
 */
app.post('/comment', (req, res) => {
  const { predictionAll, predictionTop } = req.body;

  if (predictionAll === undefined || predictionTop === undefined) {
    return res.status(400).json({ error: 'Missing predictionAll or predictionTop' });
  }

  const comments = [];

  if (predictionAll > predictionTop) {
    comments.push("Using all features slightly improved the prediction.");
  } else if (predictionAll < predictionTop) {
    comments.push("Using top 30 features gave a better prediction.");
  } else {
    comments.push("Both models predicted the same value.");
  }

  if (predictionAll > 500000) {
    comments.push("This is a high-value property. Consider double-checking luxury features.");
  } else if (predictionAll < 200000) {
    comments.push("This is a low-value estimate. Maybe the location or condition impacts price.");
  }

  return res.json({ comments });
});

// === Root check ===
app.get('/', (req, res) => {
  res.send('API is running...');
});

app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});
