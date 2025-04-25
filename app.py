"""
Dynamic-KG: Main Application Entry Point

This is the main entry point for the Dynamic-KG application, which creates
intelligent digital twins of entities and their environments through evolving knowledge graphs.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# Ensure the current directory is in the path for imports
current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

# Import the config and utilities
from config import Config
from utils.ui.styles import set_page_config, card, badge
from utils.ui.session import SessionState
from utils.data.processors import load_synthetic_data

# Initialize configuration
Config.initialize()

# Set up logger
logger = Config.get_logger("dynamic_kg_app")
logger.info("Dynamic-KG application starting")

# Set page configuration
set_page_config()

# Track page view
SessionState.track_page_view("Home")

# Title and description
st.title("Dynamic-KG 🧠")
st.markdown("""
**Dynamic-KG** creates intelligent digital twins of entities through evolving knowledge graphs.
By combining graph databases with contextual AI, we transform how insights are drawn from complex, interconnected data.
""")

# Main content
st.markdown("### Focus Areas")
col1, col2, col3 = st.columns(3)

with col1:
    card("Temporal Evolution", 
         "Knowledge representations that evolve over time, adapting to new information and changing contexts.", 
         "primary")

with col2:
    card("Contextual Intelligence", 
         "Entities understood not in isolation, but as part of a broader network of relationships.", 
         "success")

with col3:
    card("Evidence-Based Insights", 
         "Transparent reasoning with confidence scoring and clear attribution of evidence.", 
         "warning")

# Data summary
st.markdown("### Data Overview")

# Check data availability
data_available = False
company_count = len(list(Config.COMPANIES_DIR.glob("*"))) if Config.COMPANIES_DIR.exists() else 0
has_market_data = (Config.MARKET_DIR / "economic.parquet").exists() if Config.MARKET_DIR.exists() else False
has_synthetic_data = (Config.SYNTHETIC_DIR / "loan_applications.parquet").exists() if Config.SYNTHETIC_DIR.exists() else False

# Get synthetic data metrics if available
company_industries = []
if has_synthetic_data:
    data_available = True
    try:
        synthetic_companies = load_synthetic_data("companies")
        if synthetic_companies is not None and 'industry' in synthetic_companies.columns:
            company_industries = synthetic_companies['industry'].value_counts().to_dict()
    except Exception as e:
        logger.error(f"Error loading synthetic data: {e}")

# Display metrics
metric_col1, metric_col2, metric_col3 = st.columns(3)

with metric_col1:
    if company_count > 0:
        data_available = True
        st.metric("Public Companies", company_count)
    elif has_synthetic_data and synthetic_companies is not None:
        st.metric("Synthetic Companies", len(synthetic_companies))
    else:
        st.metric("Companies", "0")

with metric_col2:
    if has_market_data:
        economic_metric = "✓ Available"
    else:
        economic_metric = "✗ Missing"
    st.metric("Market Data", economic_metric)

with metric_col3:
    if has_synthetic_data:
        loan_apps = load_synthetic_data("loan_applications")
        if loan_apps is not None:
            loan_metric = str(len(loan_apps))
        else:
            loan_metric = "0"
    else:
        loan_metric = "✗ Missing"
    st.metric("Loan Applications", loan_metric)

# Get started guidance
if not data_available:
    st.warning("""
    **No data detected.** To get started:
    1. Run `python data/external/collect_fmp_data.py` to collect financial data
    2. Run `python data/synthetic/generate_loan_data.py` to create synthetic loan applications
    """)

# Navigation options 
st.markdown("### Explore the System")

option_col1, option_col2 = st.columns(2)

with option_col1:
    card("Data Explorer", """
    Browse company financials, market indicators, and synthetic loan data to understand the foundation of the knowledge graph.
    
    [Open Data Explorer](Data_Explorer)
    """, "primary")

with option_col2:
    card("Knowledge Graph Visualizer", """
    See how entities and relationships evolve over time, with confidence scoring and evidence tracking.
    
    [Open Graph Visualizer](Graph_Visualizer)
    """, "secondary")

# Footer
st.markdown("---")
st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center;">
    <span style="font-size: 0.8em;">Dynamic-KG - Experimental Knowledge Graph Framework</span>
    <span style="font-size: 0.8em;">
        <a href="https://github.com/Kris-Nale314/Dynamic-KG" target="_blank">GitHub</a> | 
        <a href="https://github.com/Kris-Nale314/Dynamic-KG/blob/main/docs/DATA_STRATEGY.md" target="_blank">Documentation</a>
    </span>
</div>
""", unsafe_allow_html=True)