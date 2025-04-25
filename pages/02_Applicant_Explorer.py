"""
02_Applicant_Explorer.py

This page provides detailed exploration of individual loan applications,
showing company information, financial data, and risk assessment.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import json

# Add the project root directory to the Python path for importing config
root_dir = str(Path(__file__).parents[1].absolute())
if root_dir not in sys.path:
    sys.path.append(root_dir)

# Import the config and utilities
from config import Config
from utils.ui.styles import set_page_config, card, badge, header_with_info
from utils.ui.session import SessionState
from utils.data.processors import load_synthetic_data

# Initialize configuration
Config.initialize()

# Set up logger
logger = Config.get_logger("loan_applicant_explorer")
logger.info("Loan Applicant Explorer page loaded")

# Set page configuration
set_page_config()

# Track page view
SessionState.track_page_view("Applicant Explorer")

# Title and description
st.title("Loan Applicant Explorer")
st.markdown("""
Explore individual loan applications in detail, including company information, 
financial data, and risk assessment. This view provides insights into the specific 
factors that influence loan approval decisions.
""")

# In the 02_Applicant_Explorer.py file, find the section that loads the data
@st.cache_data
def load_all_data():
    """Load all synthetic data required for the application explorer."""
    loans = load_synthetic_data("loan_applications")
    companies = load_synthetic_data("companies")
    owners = load_synthetic_data("owners")
    financials = load_synthetic_data("financials")
    risks = load_synthetic_data("risk_metrics")
    market = load_synthetic_data("market_conditions")
    
    # Check if data is available
    if loans is None or companies is None or risks is None:
        return None
    
    # Join industry information from companies to loans
    if 'industry' not in loans.columns and companies is not None:
        # Create a mapping from company_id to industry
        company_industry = companies[['entity_id', 'industry']].set_index('entity_id')
        # Add industry to loans using company_id
        loans = loans.copy()
        loans['industry'] = loans['company_id'].map(company_industry['industry'])
    
    return {
        "loans": loans,
        "companies": companies,
        "owners": owners,
        "financials": financials,
        "risks": risks,
        "market": market
    }

# Load the data
data = load_all_data()

if data is None:
    st.error("Required data not found. Please generate synthetic data first.")
    st.stop()

# Create a company selector in the sidebar
st.sidebar.header("Application Selection")

# Select time period
time_periods = sorted(data["loans"]["time_period"].unique())
selected_period = st.sidebar.selectbox(
    "Select Time Period", 
    time_periods,
    index=0
)

# Filter loans by time period
period_loans = data["loans"][data["loans"]["time_period"] == selected_period]

# Group applications by company and create a selection box
companies = period_loans[["company_id", "company_name"]].drop_duplicates()
company_options = companies["company_name"].tolist()
company_ids = companies["company_id"].tolist()

selected_company_name = st.sidebar.selectbox(
    "Select Company",
    company_options
)

# Get the selected company ID
selected_company_id = companies[companies["company_name"] == selected_company_name]["company_id"].iloc[0]

# Get the loan application for the selected company and time period
loan = period_loans[period_loans["company_id"] == selected_company_id].iloc[0]

# Get related data
company = data["companies"][data["companies"]["entity_id"] == selected_company_id].iloc[0]
owner = data["owners"][data["owners"]["owns_company_id"] == selected_company_id].iloc[0]
financial = data["financials"][
    (data["financials"]["company_id"] == selected_company_id) & 
    (data["financials"]["time_period"] == selected_period)
].iloc[0]
risk = data["risks"][data["risks"]["loan_id"] == loan["entity_id"]].iloc[0]
market_condition = data["market"][data["market"]["entity_id"] == loan["market_condition_id"]].iloc[0]

# Main dashboard
st.header(f"Loan Application: {selected_company_name}")

# Loan overview metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Loan Amount", f"${loan['loan_amount']:,.0f}")

with col2:
    st.metric("Interest Rate", f"{loan['interest_rate']*100:.2f}%")

with col3:
    st.metric("Term", f"{loan['term_years']} years")

with col4:
    st.metric("Purpose", loan["loan_purpose"])

# Application status
st.subheader("Application Status")
status_col1, status_col2 = st.columns([1, 3])

with status_col1:
    # Create risk gauge
    risk_score = risk["risk_score"]
    max_risk_score = 150  # Based on our synthetic data generator
    
    # Create gauge chart for risk score
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = risk_score,
        title = {'text': "Risk Score"},
        gauge = {
            'axis': {'range': [0, max_risk_score], 'tickwidth': 1},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "green"},
                {'range': [50, 100], 'color': "yellow"},
                {'range': [100, max_risk_score], 'color': "red"}
            ],
        }
    ))
    
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)

with status_col2:
    # Create confidence progress bars for approval
    approval_status = risk["approval_status"]
    approval_probability = risk["approval_probability"]
    
    st.markdown(f"**Approval Status**: {approval_status}")
    st.progress(approval_probability)
    st.markdown(f"**Approval Probability**: {approval_probability:.2%}")

    # Display application date and desired funding date
    st.markdown(f"**Application Date**: {loan['application_date']}")
    st.markdown(f"**Desired Funding Date**: {loan['desired_funding_date']}")

# Tabs for different aspects of the application
tab1, tab2, tab3, tab4 = st.tabs(["Company Profile", "Financial Data", "Risk Assessment", "Market Context"])

with tab1:
    # Company profile
    st.subheader("Company Information")
    profile_col1, profile_col2 = st.columns(2)
    
    with profile_col1:
        st.markdown(f"**Industry**: {company['industry']}")
        st.markdown(f"**Business Type**: {company['business_type']}")
        st.markdown(f"**Years in Business**: {company['years_in_business']}")
        st.markdown(f"**Employees**: {company['employees']}")
        st.markdown(f"**Annual Revenue**: ${company['annual_revenue']:,.0f}")
    
    with profile_col2:
        st.markdown(f"**Business Start Date**: {company['business_start_date']}")
        st.markdown(f"**Address**: {company['business_address']}")
        st.markdown(f"**City**: {company['business_city']}")
        st.markdown(f"**State**: {company['business_state']}")
        st.markdown(f"**ZIP**: {company['business_zip']}")
    
    # Owner information
    st.subheader("Owner Information")
    owner_col1, owner_col2 = st.columns(2)
    
    with owner_col1:
        st.markdown(f"**Owner Name**: {owner['owner_name']}")
        st.markdown(f"**Title**: {owner['owner_title']}")
        st.markdown(f"**Age**: {owner['owner_age']}")
        st.markdown(f"**Experience**: {owner['owner_experience_years']} years")
    
    with owner_col2:
        st.markdown(f"**Ownership**: {owner['ownership_percentage']}%")
        st.markdown(f"**Credit Score**: {owner['owner_credit_score']}")
        st.markdown(f"**Email**: {owner['owner_email']}")
        st.markdown(f"**Phone**: {owner['owner_phone']}")

with tab2:
    # Financial data
    st.subheader("Financial Overview")
    
    # Key financial metrics
    fin_col1, fin_col2, fin_col3 = st.columns(3)
    
    with fin_col1:
        st.metric("Annual Revenue", f"${financial['annual_revenue']:,.0f}")
        st.metric("Gross Profit", f"${financial['gross_profit']:,.0f}")
        st.metric("Net Income", f"${financial['net_income']:,.0f}")
    
    with fin_col2:
        st.metric("Total Assets", f"${financial['total_assets']:,.0f}")
        st.metric("Total Liabilities", f"${financial['total_liabilities']:,.0f}")
        st.metric("Equity", f"${financial['equity']:,.0f}")
    
    with fin_col3:
        st.metric("Current Ratio", f"{financial['current_ratio']:.2f}")
        st.metric("Debt to Equity", f"{financial['debt_to_equity']:.2f}")
        st.metric("DSCR", f"{financial['debt_service_coverage_ratio']:.2f}")
    
    # Financial ratios chart
    st.subheader("Financial Ratios")
    
    # Create a radar chart of key ratios
    ratios = {
        "Gross Margin": financial["gross_margin"],
        "Net Margin": financial["net_margin"],
        "Current Ratio": min(financial["current_ratio"], 3) / 3,  # Normalize to 0-1
        "DSCR": min(financial["debt_service_coverage_ratio"], 3) / 3,  # Normalize to 0-1
        "D/E Ratio Inv.": 1 / (financial["debt_to_equity"] + 0.1)  # Inverse and normalize
    }
    
    # Prepare the data for the radar chart
    categories = list(ratios.keys())
    values = list(ratios.values())
    
    # Add the first point at the end to close the polygon
    categories.append(categories[0])
    values.append(values[0])
    
    # Create the radar chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Financial Ratios'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    # Risk assessment
    st.subheader("Risk Factor Analysis")
    
    # Extract risk factors
    risk_factors = {}
    for key, value in risk.items():
        if key.endswith("_risk") and isinstance(value, str):
            factor_name = key.replace("_risk", "").replace("_", " ").title()
            risk_factors[factor_name] = value
    
    # Display risk factors
    for factor, level in risk_factors.items():
        # Determine color based on risk level
        if level == "High":
            color = "red"
        elif level == "Medium":
            color = "orange"
        elif level == "Low":
            color = "yellow"
        else:
            color = "green"
        
        # Create a bar with color based on risk level
        st.markdown(f"**{factor}**: <span style='color:{color};'>{level}</span>", unsafe_allow_html=True)
        
        # Add a progress bar to visualize risk level
        if level == "High":
            st.progress(0.9)
        elif level == "Medium":
            st.progress(0.6)
        elif level == "Low":
            st.progress(0.3)
        else:
            st.progress(0.1)
    
    # Confidence analysis
    st.subheader("Data Confidence")
    
    # Display confidence score
    confidence = loan.get("confidence", 0.8)  # Default if not available
    st.markdown(f"**Overall Confidence Score**: {confidence:.2f}")
    st.progress(confidence)
    
    # Display evidence attribution
    st.markdown("**Evidence Source**: " + loan.get("source", "Synthetic Data"))
    
    if "evidence_id" in loan:
        st.markdown(f"**Evidence ID**: {loan['evidence_id']}")
    
    # Display temporal attributes
    st.markdown("**Temporal Validity**")
    if "valid_from" in loan:
        st.markdown(f"**Valid From**: {loan['valid_from']}")
    
    if "valid_to" in loan:
        valid_to = loan["valid_to"] if loan["valid_to"] else "Present"
        st.markdown(f"**Valid To**: {valid_to}")

with tab4:
    # Market context
    st.subheader("Market Environment")
    
    # Display market indicators
    market_col1, market_col2 = st.columns(2)
    
    with market_col1:
        st.metric("Market Interest Rate", f"{market_condition['interest_rate']*100:.2f}%")
        st.metric("Inflation Rate", f"{market_condition['inflation_rate']*100:.2f}%")
        st.metric("Market Stability", f"{market_condition['market_stability']:.2f}")
    
    with market_col2:
        st.metric("Unemployment Rate", f"{market_condition['unemployment_rate']*100:.2f}%")
        st.metric("GDP Growth Rate", f"{market_condition['gdp_growth_rate']*100:.2f}%")
    
    # Industry-specific context
    st.subheader("Industry Context")
    
    # Get industry conditions
    industry = company["industry"]
    if "industry_conditions" in market_condition and industry in market_condition["industry_conditions"]:
        industry_data = market_condition["industry_conditions"][industry]
        
        # Display industry metrics
        ind_col1, ind_col2 = st.columns(2)
        
        with ind_col1:
            st.metric("Industry Growth Rate", f"{industry_data.get('growth_rate', 0)*100:.2f}%")
            st.metric("Industry Risk Score", f"{industry_data.get('risk_score', 0):.2f}")
        
        with ind_col2:
            st.metric("Industry Sentiment", industry_data.get("sentiment", "Neutral").title())
            st.metric("Industry Trend", industry_data.get("trend", "Stable").title())
    else:
        st.info(f"No specific industry data available for {industry}")
    

    # Visualize how this application compares to market averages
    st.subheader("Market Comparison")

    # Get all loans in the same time period and industry
    if 'industry' in period_loans.columns:
        # Get the industry from the current loan/company
        industry = company["industry"]
        industry_loans = period_loans[period_loans["industry"] == industry]
        
        if len(industry_loans) > 1:
            # Calculate industry averages
            avg_interest_rate = industry_loans["interest_rate"].mean()
            avg_loan_amount = industry_loans["loan_amount"].mean()
            
            # Create comparison chart
            comparison_data = pd.DataFrame({
                "Category": ["Interest Rate", "Loan Amount"],
                "This Application": [loan["interest_rate"], loan["loan_amount"] / avg_loan_amount],
                "Industry Average": [avg_interest_rate, 1.0]  # Normalized loan amount
            })
            
            # Reshape for plotting
            comparison_melted = pd.melt(
                comparison_data, 
                id_vars=["Category"],
                var_name="Comparison",
                value_name="Value"
            )
            
            # Create the chart
            fig = px.bar(
                comparison_melted,
                x="Category",
                y="Value",
                color="Comparison",
                barmode="group",
                title="Comparison to Industry Average",
                labels={"Value": "Relative Value", "Category": "Metric"}
            )
            
            # Adjust y-axis for different scales
            fig.update_layout(
                yaxis=dict(
                    tickformat=".2%"
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data to compare with industry averages")
    else:
        st.info("Industry information not available for comparison")

# Footer
st.markdown("---")
st.markdown("""
This page demonstrates how the knowledge graph integrates information about a loan application with
its broader context, including company profile, financial performance, and market environment.
""")