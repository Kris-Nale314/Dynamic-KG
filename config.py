"""
Configuration management for the Dynamic-KG project.

This module handles all configuration settings, including paths, API credentials,
and application parameters for this POC/MVP implementation.
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define root directory and ensure it's in the Python path
ROOT_DIR = Path(__file__).parent.absolute()
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

class Config:
    """Configuration class for the Dynamic-KG POC/MVP."""
    
    # Application paths
    ROOT_DIR = ROOT_DIR
    CORE_DIR = ROOT_DIR / "core"
    
    # DataStore structure - simplified for POC
    DATASTORE_DIR = ROOT_DIR / os.getenv("DATASTORE_DIR", "dataStore")
    COMPANIES_DIR = DATASTORE_DIR / "companies"
    MARKET_DIR = DATASTORE_DIR / "market"
    EVENTS_DIR = DATASTORE_DIR / "events"
    RELATIONSHIPS_DIR = DATASTORE_DIR / "relationships"
    SYNTHETIC_DIR = DATASTORE_DIR / "synthetic"
    KG_DATA_DIR = DATASTORE_DIR / "knowledge_graph"
    
    # Output directories
    OUTPUT_DIR = ROOT_DIR / os.getenv("OUTPUT_DIR", "outputs")
    VISUALIZATION_DIR = OUTPUT_DIR / "visualizations"
    
    # Logs
    LOGS_DIR = ROOT_DIR / os.getenv("LOGS_DIR", "logs")
    LOG_FILE = LOGS_DIR / f"dynamic_kg_{datetime.now().strftime('%Y%m%d')}.log"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # Neo4j settings (Community Edition)
    NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
    
    # FMP API settings
    FMP_API_KEY = os.getenv("FMP_API_KEY")
    FMP_API_BASE_URL = os.getenv("FMP_API_BASE_URL", "https://financialmodelingprep.com/api/v3")
    
    # LLM settings
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    
    # Knowledge Graph settings - core to our innovation
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
    TEMPORAL_PROPERTY_PREFIX = os.getenv("TEMPORAL_PROPERTY_PREFIX", "valid_")
    
    # Source reliability settings - simplified for POC
    SOURCE_RELIABILITY = {
        "fmp_api": 0.9,
        "sec_filing": 0.95,
        "news_article": 0.7,
        "synthetic": 0.8
    }
    
    # Synthetic data generation
    SYNTHETIC_COMPANY_COUNT = int(os.getenv("SYNTHETIC_COMPANY_COUNT", "20"))  # Smaller for POC
    SYNTHETIC_TIME_PERIODS = int(os.getenv("SYNTHETIC_TIME_PERIODS", "4"))    # Fewer periods for POC
    
    # Streamlit settings
    STREAMLIT_THEME = os.getenv("STREAMLIT_THEME", "dark")
    
    @classmethod
    def initialize(cls):
        """Create necessary directories if they don't exist and set up logging."""
        # Create directories
        for dir_path in [
            cls.DATASTORE_DIR, 
            cls.COMPANIES_DIR, 
            cls.MARKET_DIR, 
            cls.EVENTS_DIR, 
            cls.RELATIONSHIPS_DIR, 
            cls.SYNTHETIC_DIR, 
            cls.KG_DATA_DIR,
            cls.OUTPUT_DIR, 
            cls.VISUALIZATION_DIR, 
            cls.LOGS_DIR
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        logging.basicConfig(
            level=getattr(logging, cls.LOG_LEVEL),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(cls.LOG_FILE),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        # Log initialization
        logging.info(f"Dynamic-KG POC initialized with root directory: {cls.ROOT_DIR}")
        
        # Check for required API keys
        if not cls.FMP_API_KEY:
            logging.warning("FMP_API_KEY not set. External data integration will be limited.")
        
        if not cls.OPENAI_API_KEY:
            logging.warning("OPENAI_API_KEY not set. LLM functionality will be disabled.")
    
    @classmethod
    def get_logger(cls, name):
        """Get a logger with the specified name."""
        return logging.getLogger(name)
    
    @classmethod
    def get_company_dir(cls, ticker):
        """Get the directory for a specific company."""
        company_dir = cls.COMPANIES_DIR / ticker.upper()
        company_dir.mkdir(parents=True, exist_ok=True)
        return company_dir


# Initialize on import
if os.getenv("INITIALIZE_ON_IMPORT", "True").lower() == "true":
    Config.initialize()