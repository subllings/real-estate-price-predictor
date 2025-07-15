/**
 * Demo Guide - How to use the Global Admin Panel for presentations
 */

# 🎯 Admin Panel Demo Guide

## Global Access
Le panneau d'administration est maintenant accessible depuis **toutes les pages** de l'application, parfait pour tes présentations !

## ⌨️ Raccourcis Clavier
- **Ctrl + A** : Ouvre/Ferme le panneau admin
- **Ctrl + M** : Focus sur l'onglet Models 
- **Ctrl + T** : Focus sur l'onglet Training
- **Ctrl + D** : Focus sur l'onglet Documents
- **Ctrl + R** : Focus sur l'onglet Monitoring

## 📋 Scenarios de Démo

### 1. Real Estate Price Predictor + Admin
```
1. Aller sur la page principale (/)
2. Saisir une propriété à évaluer
3. Appuyer sur Ctrl+A pour ouvrir l'admin
4. Montrer les métriques en temps réel pendant la prédiction
5. Switcher vers l'onglet Models pour montrer les performances A/B
```

### 2. ESG Agent + Document Management
```
1. Aller sur /esg-agent 
2. Poser une question au chatbot
3. Ctrl+A → Onglet Documents
4. Montrer les documents RAG utilisés pour la réponse
5. Upload d'un nouveau document en live
```

### 3. Training en Cours (Azure ML)
```
1. Sur n'importe quelle page
2. Ctrl+A → Onglet Training
3. Lancer un nouveau training Azure ML
4. Montrer le progress en temps réel
5. Validation des quality gates (R² ≥ 0.85)
```

## 🎨 Interface Elements

### Floating Button
- Position: Fixed bottom-right (25px from edges)
- Style: Blue gradient with pulse animation
- Icon: ⚙️ Settings gear
- Hover: Scale + shadow effect

### Sliding Panel
- Width: 400px (responsive sur mobile)
- Animation: Slide from right with backdrop blur
- Style: Modern glass-morphism effect
- Z-index: 9999 (au-dessus de tout)

### Tabs System
- **Models** 📊 : Model performance, A/B testing, promotion
- **Training** 🚀 : Azure ML jobs, progress, quality gates  
- **Documents** 📄 : RAG uploads, vector store status
- **Monitoring** 📈 : Real-time metrics, system health

## 🚀 Demo Flow Suggestions

### Scenario 1: "Démo Technique Complète"
1. **Intro** (Page principale)
   - Présenter l'interface utilisateur
   - Montrer une prédiction de prix
   
2. **Behind the Scenes** (Ctrl+A)
   - Ouvrir l'admin pendant que la prédiction se fait
   - Montrer les métriques temps réel
   - Expliquer l'architecture ML
   
3. **Model Management**
   - Onglet Models → A/B testing
   - Production vs Candidate models
   - Métriques de performance
   
4. **Training Pipeline**
   - Onglet Training → Azure ML
   - Lancer un entraînement en live
   - Quality gates et monitoring

### Scenario 2: "Focus Business Intelligence"
1. **ESG Agent Demo**
   - Poser des questions business
   - Montrer les réponses intelligentes
   
2. **Document Intelligence** (Ctrl+A → Documents)
   - Montrer la base de connaissances
   - Upload de nouvelles données
   - Impact en temps réel sur les réponses

### Scenario 3: "Technical Deep Dive"
1. **Architecture Overview**
   - Admin → Monitoring
   - Infrastructure metrics
   - Performance monitoring
   
2. **ML Pipeline** 
   - Training tab → Azure ML integration
   - Montrer le cloud training
   - Quality assurance automatique

## 📊 Key Metrics to Highlight

### Model Performance
- **R² Score**: ≥ 0.85 (quality gate)
- **Validation Gap**: ≤ 0.05 (no overfitting)
- **Inference Time**: < 200ms
- **Throughput**: 1000+ predictions/min

### Training Success
- **Azure ML Compute**: Tesla V100 GPUs
- **Auto-scaling**: 0-4 nodes
- **Cost Efficiency**: €62/month vs local crashes
- **Quality Gates**: Automated validation

### System Health
- **API Response Time**: < 100ms
- **Uptime**: 99.9%
- **Vector Search**: < 50ms latency
- **Document Processing**: Real-time

## 🎯 Key Demo Messages

1. **"Seamless Integration"** 
   → Admin accessible partout, pas de context switching

2. **"Enterprise-Grade Monitoring"**
   → Métriques temps réel, quality gates automatiques

3. **"Cloud-Native Architecture"**
   → Azure ML, scalabilité automatique, coûts maîtrisés

4. **"Intelligence Augmentée"**
   → RAG system, documents business, IA contextuelle

## 💡 Pro Tips for Demo

- **Préparer les données** : Avoir des documents pré-uploadés intéressants
- **Simuler l'activité** : Quelques prédictions en background pour des métriques vivantes  
- **Raccourcis clavier** : Impressioner avec Ctrl+A fluide
- **Storytelling** : Partir de l'interface utilisateur vers la technique
- **Interactive** : Laisser l'audience suggérer des tests

## 🔧 Technical Setup for Demo

### Before Demo
```bash
# Ensure all services are running
npm run dev  # React frontend
python backend-api-price-prediction/main.py  # Price API
python backend-api-llm-v2/main.py  # LLM API

# Optional: Pre-populate with demo data
curl -X POST localhost:8000/admin/demo-data
```

### During Demo
- Keep terminal open for potential debugging
- Have Azure ML workspace ready
- Prepare interesting documents for upload
- Test keyboard shortcuts beforehand

---

**Résultat** : Présentation fluide et professionnelle avec accès admin transparent depuis toutes les pages ! 🎯
