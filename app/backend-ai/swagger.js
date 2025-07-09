// swagger.js
const swaggerJSDoc = require('swagger-jsdoc');
const swaggerUi = require('swagger-ui-express');

const options = {
  definition: {
    openapi: '3.0.0',
    info: {
      title: 'Azure OpenAI API',
      version: '1.0.0',
      description: 'Simple API to interact with Azure OpenAI using Node.js',
    },
  },
  apis: ['./index.js'], // Or another file where you define your routes
};

const swaggerSpec = swaggerJSDoc(options);

function setupSwagger(app) {
  app.use('/docs', swaggerUi.serve, swaggerUi.setup(swaggerSpec));
}

module.exports = setupSwagger;
