#!/usr/bin/env python3
"""
Setup script for cloud-enhanced real estate model training
This script helps you get started with the improved architecture
"""

import os
import sys
import subprocess
from pathlib import Path


def print_header(text):
    print("\n" + "="*60)
    print(f" {text}")
    print("="*60)


def print_step(step, text):
    print(f"\n📋 Step {step}: {text}")


def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version.split()[0]} is compatible")
    return True


def check_virtual_env():
    """Check if running in virtual environment"""
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Running in virtual environment")
        return True
    else:
        print("⚠️  Not running in virtual environment")
        print("   Consider activating your .venv: source .venv/Scripts/activate")
        return False


def install_dependencies():
    """Install required packages"""
    try:
        # Install requirements from the existing requirements.txt
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements from requirements.txt installed")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        print("🔧 Try installing manually with: pip install -r requirements.txt")
        return False


def setup_environment():
    """Setup environment configuration"""
    env_template = Path(".env.template")
    env_file = Path(".env")
    
    if env_template.exists() and not env_file.exists():
        print("📄 Creating .env file from template...")
        with open(env_template, 'r') as template:
            content = template.read()
        
        with open(env_file, 'w') as env:
            env.write(content)
        
        print("✅ .env file created")
        print("⚠️  Please edit .env file with your Azure credentials")
        return True
    elif env_file.exists():
        print("✅ .env file already exists")
        return True
    else:
        print("⚠️  No .env.template found, creating basic .env...")
        basic_env = """# Azure Configuration
AZURE_STORAGE_CONNECTION_STRING=""
AZURE_COSMOS_CONNECTION_STRING=""
AZURE_OPENAI_API_KEY=""
ENVIRONMENT="development"
"""
        with open(env_file, 'w') as f:
            f.write(basic_env)
        print("✅ Basic .env file created")
        return True


def create_directories():
    """Create necessary directories"""
    directories = [
        "ml_models",
        "reports", 
        "configs",
        "agents/__pycache__",
        "data/temp"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    print("✅ All directories created")


def test_imports():
    """Test if all required imports work"""
    print("🧪 Testing imports...")
    
    required_modules = [
        ("pandas", "pd"),
        ("numpy", "np"),
        ("sklearn.model_selection", "train_test_split"),
        ("catboost", "CatBoostRegressor"),
        ("optuna", None),
        ("json", None),
        ("pickle", None)
    ]
    
    optional_modules = [
        ("azure.storage.blob", "BlobServiceClient"),
        ("azure.cosmos", "CosmosClient")
    ]
    
    all_good = True
    
    # Test required modules
    for module, item in required_modules:
        try:
            if item:
                exec(f"from {module} import {item}")
            else:
                exec(f"import {module}")
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            all_good = False
    
    # Test optional modules
    print("\n🔧 Optional (Azure) modules:")
    for module, item in optional_modules:
        try:
            if item:
                exec(f"from {module} import {item}")
            else:
                exec(f"import {module}")
            print(f"✅ {module}")
        except ImportError:
            print(f"⚠️  {module} (will use local-only mode)")
    
    return all_good


def run_basic_test():
    """Run a basic test of the training system"""
    print("🚀 Running basic test...")
    
    try:
        # Import and test the training agent
        from agents.cloud_training_agent import CloudTrainingAgent
        
        # Create agent instance
        agent = CloudTrainingAgent()
        print("✅ CloudTrainingAgent created successfully")
        
        # Test configuration loading
        config = agent.config
        print(f"✅ Configuration loaded: {len(config)} sections")
        
        # Test model directory
        if agent.local_models_dir.exists():
            print(f"✅ Models directory: {agent.local_models_dir}")
        
        # Test cloud connection (if credentials available)
        if agent.blob_client:
            print("✅ Azure Blob Storage client initialized")
        else:
            print("ℹ️  Azure Blob Storage not configured (local-only mode)")
        
        if agent.cosmos_client:
            print("✅ Azure Cosmos DB client initialized")
        else:
            print("ℹ️  Azure Cosmos DB not configured (local-only mode)")
        
        print("✅ Basic test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Basic test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def show_next_steps():
    """Show what to do next"""
    print_header("🎯 NEXT STEPS")
    
    print("""
1. 📝 Configure Azure (optional):
   - Edit .env file with your Azure credentials
   - Create Azure Storage Account and Cosmos DB
   
2. 🚀 Run enhanced training:
   - python train_with_cloud_sync.py
   
3. 🔧 Test model management:
   - Check ml_models/ directory for saved models
   - View reports/ directory for training reports
   
4. 🌐 Integrate with your API:
   - Models will be automatically copied to API directory
   - Use cloud_model_manager.py for enhanced model management
   
5. 📊 Monitor and improve:
   - Set up Azure dashboards
   - Implement automated retraining
   - Add performance monitoring

📚 Documentation:
   - Check configs/training_config.json for settings
   - Read agents/cloud_training_agent.py for architecture details
   - Review .env.template for all configuration options
""")


def main():
    """Main setup function"""
    print_header("🏠 Real Estate AI Training Setup")
    
    # Step 1: Check Python version
    print_step(1, "Checking Python version")
    if not check_python_version():
        return False
    
    # Step 2: Check virtual environment
    print_step(2, "Checking virtual environment")
    check_virtual_env()
    
    # Step 3: Create directories
    print_step(3, "Creating directories")
    create_directories()
    
    # Step 4: Setup environment
    print_step(4, "Setting up environment")
    setup_environment()
    
    # Step 5: Install dependencies
    print_step(5, "Installing dependencies")
    if not install_dependencies():
        print("❌ Setup failed due to dependency issues")
        return False
    
    # Step 6: Test imports
    print_step(6, "Testing imports")
    if not test_imports():
        print("⚠️  Some imports failed, but you can still use local-only mode")
    
    # Step 7: Run basic test
    print_step(7, "Running basic test")
    if not run_basic_test():
        print("⚠️  Basic test failed, but setup is complete")
    
    # Show next steps
    show_next_steps()
    
    print("\n🎉 Setup completed! You can now run enhanced model training.")
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⛔ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
