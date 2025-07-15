"""
Azure ML Setup Script for Real Estate Price Predictor
Configures Azure ML workspace and compute for cloud training
"""

import os
import json
from pathlib import Path

# Azure ML imports
try:
    from azureml.core import Workspace
    from azureml.core.compute import ComputeTarget, AmlCompute
    from azureml.core.compute_target import ComputeTargetException
    from azureml.exceptions import WorkspaceException
    AZURE_ML_AVAILABLE = True
except ImportError:
    print("❌ Azure ML SDK not installed. Install with:")
    print("pip install azureml-sdk[complete]")
    AZURE_ML_AVAILABLE = False


def setup_azure_ml_workspace():
    """
    Setup Azure ML workspace configuration
    
    Prerequisites:
    1. Azure subscription
    2. Resource group created
    3. ML workspace created in Azure Portal
    4. config.json downloaded from workspace
    """
    
    print("🔧 Setting up Azure ML Workspace...")
    
    # Check if config.json exists
    config_path = Path("config.json")
    if not config_path.exists():
        print("❌ config.json not found!")
        print("📝 Please:")
        print("1. Go to Azure Portal → Your ML Workspace")
        print("2. Click 'Download config.json'")
        print("3. Place it in the project root")
        return None
    
    try:
        # Connect to workspace
        ws = Workspace.from_config()
        print(f"✅ Connected to workspace: {ws.name}")
        print(f"📍 Location: {ws.location}")
        print(f"🔗 Resource Group: {ws.resource_group}")
        
        return ws
        
    except WorkspaceException as e:
        print(f"❌ Failed to connect to workspace: {e}")
        return None


def create_compute_cluster(workspace, cluster_name="gpu-cluster"):
    """
    Create GPU compute cluster for training
    
    Args:
        workspace: Azure ML workspace
        cluster_name: Name for the compute cluster
    """
    
    print(f"🖥️  Setting up compute cluster: {cluster_name}")
    
    try:
        # Check if cluster already exists
        compute_target = ComputeTarget(workspace=workspace, name=cluster_name)
        print(f"✅ Compute cluster '{cluster_name}' already exists")
        
    except ComputeTargetException:
        print(f"🔨 Creating new compute cluster: {cluster_name}")
        
        # Compute configuration
        compute_config = AmlCompute.provisioning_configuration(
            vm_size='Standard_NC6s_v3',  # Tesla V100 GPU
            min_nodes=0,                 # Scale to zero when idle
            max_nodes=2,                 # Max 2 nodes for cost control
            idle_seconds_before_scaledown=300,  # 5min idle → scale down
            tier='Dedicated',            # Better performance
            description='GPU cluster for real estate model training'
        )
        
        # Create cluster
        compute_target = ComputeTarget.create(workspace, cluster_name, compute_config)
        compute_target.wait_for_completion(show_output=True)
        
        print(f"✅ Compute cluster '{cluster_name}' created successfully")
    
    return compute_target


def create_training_environment(workspace):
    """
    Create conda environment for training
    """
    
    print("📦 Setting up training environment...")
    
    # Create environment directory
    env_dir = Path("environments")
    env_dir.mkdir(exist_ok=True)
    
    # Training environment YAML
    env_yaml = """
name: realestate-training
dependencies:
  - python=3.10
  - pip
  - pip:
    - catboost>=1.2
    - optuna>=3.0
    - scikit-learn>=1.3
    - pandas>=2.0
    - numpy>=1.24
    - joblib>=1.3
    - azureml-mlflow
    - azure-storage-blob
    - azure-cosmos
"""
    
    env_file = env_dir / "training_env.yml"
    with open(env_file, 'w') as f:
        f.write(env_yaml)
    
    print(f"✅ Environment file created: {env_file}")
    
    return env_file


def create_training_config():
    """
    Create training configuration for Azure ML
    """
    
    print("⚙️  Creating training configuration...")
    
    config = {
        "model": {
            "type": "catboost",
            "target_r2": 0.85,
            "max_validation_gap": 0.05
        },
        "optuna": {
            "n_trials": 50,
            "direction": "maximize",
            "study_name": "real_estate_azure_ml"
        },
        "hyperparameters": {
            "learning_rate": [0.01, 0.3],
            "depth": [4, 10],
            "l2_leaf_reg": [1, 10],
            "iterations": [500, 2000]
        },
        "data": {
            "train_path": "data/ml_ready/train_data.parquet",
            "test_path": "data/ml_ready/test_data.parquet",
            "features_path": "configs/feature_mapping.yaml"
        },
        "quality_gates": {
            "min_r2_test": 0.85,
            "max_validation_gap": 0.05,
            "min_samples": 1000
        }
    }
    
    config_dir = Path("configs")
    config_dir.mkdir(exist_ok=True)
    
    config_file = config_dir / "azure_ml_training.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Training config created: {config_file}")
    
    return config_file


def create_submission_script():
    """
    Create script to submit training jobs to Azure ML
    """
    
    print("📝 Creating job submission script...")
    
    script_content = '''"""
Submit training job to Azure ML
"""

from azureml.core import Workspace, Experiment, ScriptRunConfig
from azureml.core.environment import Environment
from pathlib import Path

def submit_training_job():
    """Submit real estate training to Azure ML"""
    
    # Connect to workspace
    ws = Workspace.from_config()
    
    # Create experiment
    experiment = Experiment(workspace=ws, name='real-estate-cloud-training')
    
    # Load environment
    env = Environment.from_conda_specification(
        name='realestate-training',
        file_path='environments/training_env.yml'
    )
    
    # Configure training run
    config = ScriptRunConfig(
        source_directory='azure_ml/',
        script='train_azure.py',
        arguments=[
            '--config', 'configs/azure_ml_training.json',
            '--output_dir', './outputs',
        ],
        compute_target='gpu-cluster',
        environment=env
    )
    
    # Submit job
    run = experiment.submit(config)
    
    print(f"🚀 Training job submitted!")
    print(f"📊 Monitor at: {run.get_portal_url()}")
    print(f"🆔 Run ID: {run.id}")
    
    return run

if __name__ == "__main__":
    run = submit_training_job()
    
    # Optional: Wait for completion
    print("⏳ Waiting for training to complete...")
    run.wait_for_completion(show_output=True)
    
    # Get results
    metrics = run.get_metrics()
    print(f"📊 Final Results:")
    print(f"   R² Test: {metrics.get('r2_test', 'N/A')}")
    print(f"   Validation Gap: {metrics.get('validation_gap', 'N/A')}")
    print(f"   Quality Gate: {'✅ PASSED' if metrics.get('quality_gate_passed') else '❌ FAILED'}")
'''
    
    script_file = Path("submit_azure_training.py")
    with open(script_file, 'w') as f:
        f.write(script_content)
    
    print(f"✅ Submission script created: {script_file}")
    
    return script_file


def main():
    """
    Complete Azure ML setup process
    """
    
    print("🚀 Azure ML Setup for Real Estate Price Predictor")
    print("=" * 60)
    
    if not AZURE_ML_AVAILABLE:
        return
    
    # 1. Setup workspace
    workspace = setup_azure_ml_workspace()
    if not workspace:
        return
    
    # 2. Create compute cluster
    compute = create_compute_cluster(workspace)
    
    # 3. Create environment
    env_file = create_training_environment(workspace)
    
    # 4. Create training config
    config_file = create_training_config()
    
    # 5. Create submission script
    submit_script = create_submission_script()
    
    print("\n✅ Azure ML setup complete!")
    print("\n📋 Next steps:")
    print("1. Ensure your data is in data/ml_ready/ folder")
    print("2. Run: python submit_azure_training.py")
    print("3. Monitor training in Azure ML Studio")
    print("4. Download trained model when complete")
    
    print(f"\n🔗 Workspace URL: {workspace.get_details()['workspaceUrl']}")


if __name__ == "__main__":
    main()
