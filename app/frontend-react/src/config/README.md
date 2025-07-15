# API Configuration

Ce fichier gère automatiquement les URLs d'API en fonction de l'environnement :

## Environnements

### Développement (Local)
- **NODE_ENV**: `development` ou non défini
- **APIs**: 
  - Prédiction: `http://127.0.0.1:8000`
  - LLM: `http://127.0.0.1:8010`
  - Chat: `http://127.0.0.1:8010/chat`
  - Commentaires: `http://127.0.0.1:8010/comment`

### Production (Azure)
- **NODE_ENV**: `production`
- **APIs**:
  - Prédiction: `https://realestate-api.azurewebsites.net`
  - LLM: `https://realestate-api-llm-v2.azurewebsites.net`
  - Chat: `https://realestate-api-llm-v2.azurewebsites.net/chat`
  - Commentaires: `https://realestate-api-llm-v2.azurewebsites.net/comment`

## Usage

```javascript
import { PREDICTION_API_URL, CHAT_API_URL, COMMENT_API_URL } from '../../config/api';

// Utilisation dans les composants
const response = await axios.post(PREDICTION_API_URL + '/predict_all', data);
```

## Variables d'Environnement

Pour forcer l'environnement de production en local, définir :
```bash
NODE_ENV=production npm start
```

## Fichiers Mis à Jour

- `src/components/PropertyForm/index.js`
- `src/components/SidePanel/SidePanel.jsx`
