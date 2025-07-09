require('dotenv').config();
const express = require('express');
const axios = require('axios');
const bodyParser = require('body-parser');
const cors = require('cors');
const swaggerUi = require('swagger-ui-express');
const swaggerSpecs = require('./swagger');

const app = express();
const port = process.env.PORT || 5050;

app.use(cors());
app.use(bodyParser.json());

// Serve Swagger docs at /docs
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

// Root health check
app.get('/', (req, res) => {
  res.send('API is running...');
});

app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});
