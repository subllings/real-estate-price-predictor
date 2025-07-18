"""
Configuration centralisée pour les logs Azure
Réduit la verbosité des logs Azure SDK
"""

import logging

def configure_azure_logging():
    """Configure les niveaux de logging pour Azure SDK"""
    # Réduire la verbosité des logs Azure
    logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
    logging.getLogger("azure.core.pipeline").setLevel(logging.WARNING)
    logging.getLogger("azure.cosmos").setLevel(logging.WARNING)
    logging.getLogger("azure.storage").setLevel(logging.WARNING)
    logging.getLogger("azure").setLevel(logging.WARNING)
    
    # Configuration des autres services verbeux
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    print("✅ Configuration des logs Azure appliquée - verbosité réduite")

# Appel automatique à l'import
configure_azure_logging()
