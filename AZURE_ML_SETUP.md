# Azure ML Quick Start Guide

## 🚀 Setup Instructions (15 minutes)

### 1. Azure Subscription Setup

#### Create Resources in Azure Portal:
```bash
# 1. Create Resource Group
Name: rg-realestate-ml-training
Location: West Europe

# 2. Create Machine Learning Workspace
Name: aml-realestate-predictor
Resource Group: rg-realestate-ml-training
Storage: Auto-create
Key Vault: Auto-create
Application Insights: Auto-create
```

#### Download Configuration:
1. Go to Azure Portal → Machine Learning → Your Workspace
2. Click "Download config.json"
3. Place `config.json` in project root

### 2. Local Setup

```bash
# Install Azure ML SDK
pip install azureml-sdk[complete]

# Install additional dependencies
pip install optuna mlflow

# Run setup script
python azure_ml_setup.py
```

### 3. First Training Job

```bash
# Submit training to Azure ML
python submit_azure_training.py

# Monitor in Azure ML Studio
# URL will be displayed in console
```

## 📊 Expected Results

### Training Configuration:
- **Target R²**: ≥ 0.85
- **Max Validation Gap**: ≤ 0.05
- **Parallel Trials**: 4 simultaneous
- **Compute**: Tesla V100 GPU

### Cost Estimation:
- **Setup**: Free
- **Training**: ~€3-6 per session (10-15 minutes)
- **Monthly**: ~€60-80 for regular training

### Quality Gates:
✅ R² Test ≥ 0.85  
✅ Validation Gap ≤ 0.05  
✅ No severe overfitting  
✅ Stable across runs  

## 🔍 Monitoring

### Azure ML Studio:
- Real-time training progress
- Hyperparameter comparison
- Model metrics visualization
- Cost tracking

### Local Monitoring:
```python
# Check job status
python -c "
from azureml.core import Workspace, Experiment
ws = Workspace.from_config()
exp = Experiment(ws, 'real-estate-cloud-training')
runs = list(exp.get_runs())
print(f'Latest run: {runs[0].get_status()}')
"
```

## ⚡ Benefits vs Local Training

| Aspect | Local (GTX 2080 Ti) | Azure ML Cloud |
|--------|-------------------|----------------|
| **Stability** | ❌ Crashes | ✅ Reliable |
| **Speed** | 2+ hours | 10-15 minutes |
| **Parallel** | ❌ Sequential | ✅ 4 simultaneous |
| **Development** | ❌ Blocked | ✅ Laptop free |
| **Quality** | ❌ Limited | ✅ R² ≥ 0.85 target |

## 🛠️ Troubleshooting

### Common Issues:

1. **config.json not found**
   - Download from Azure ML workspace
   - Place in project root

2. **Compute cluster creation fails**
   - Check quota limits in subscription
   - Try different VM size

3. **Training job fails**
   - Check data paths in config
   - Verify environment dependencies

4. **Quality gates not met**
   - Increase n_trials in config
   - Check data quality
   - Adjust hyperparameter ranges

## 📈 Next Steps

1. **Week 1**: Basic setup + first successful training
2. **Week 2**: Automated deployment pipeline
3. **Week 3**: Production monitoring
4. **Week 4**: Cost optimization + scaling

## 🔗 Useful Links

- [Azure ML Documentation](https://docs.microsoft.com/azure/machine-learning/)
- [Pricing Calculator](https://azure.microsoft.com/pricing/calculator/)
- [VM Sizes and Pricing](https://docs.microsoft.com/azure/virtual-machines/sizes-gpu)
