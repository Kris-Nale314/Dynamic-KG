"""
01_Portfolio_Explorer.py

This page provides a high-level view of the loan portfolio,
showing aggregate metrics, distributions, and trends.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

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
logger = Config.get_logger("loan_portfolio_explorer")
logger.info("Loan Portfolio Explorer page loaded")

# Set page configuration
set_page_config()

# Track page view
SessionState.track_page_view("Portfolio Explorer")

# Title and description
st.title("Loan Portfolio Explorer")
st.markdown("""
Explore the loan portfolio at an aggregate level, with insights into distribution by industry,
risk category, and loan purpose. This view helps understand the overall composition and health
of the loan book.
""")

# Load data
@st.cache_data
def load_portfolio_data():
    """Load all synthetic data required for the portfolio explorer."""
    loans = load_synthetic_data("loan_applications")
    companies = load_synthetic_data("companies")
    risks = load_synthetic_data("risk_metrics")
    market = load_synthetic_data("market_conditions")
    
    # Check if data is available
    if loans is None or risks is None:
        return None
    
    # Merge loans with company data if available
    if companies is not None:
        loans = pd.merge(
            loans,
            companies[["entity_id", "industry", "annual_revenue", "employees", "years_in_business"]],
            left_on="company_id",
            right_on="entity_id",
            suffixes=("", "_company")
        )
    
    # Merge loans with risk data
    loans = pd.merge(
        loans,
        risks[["loan_id", "risk_score", "approval_status", "approval_probability"]],
        left_on="entity_id",
        right_on="loan_id",
        how="left"
    )
    
    return {
        "loans": loans,
        "market": market
    }

# Load the data
data = load_portfolio_data()

if data is None:
    st.error("Required data not found. Please generate synthetic data first.")
    st.stop()

# Sidebar filters
st.sidebar.header("Portfolio Filters")

# Time period selector
time_periods = sorted(data["loans"]["time_period"].unique())
selected_period = st.sidebar.selectbox(
    "Select Time Period", 
    time_periods,
    index=0
)

# Industry filter
industries = sorted(data["loans"]["industry"].unique())
selected_industries = st.sidebar.multiselect(
    "Filter by Industry",
    industries,
    default=industries
)

# Loan purpose filter
purposes = sorted(data["loans"]["loan_purpose"].unique())
selected_purposes = st.sidebar.multiselect(
    "Filter by Loan Purpose",
    purposes,
    default=purposes
)

# Loan amount range
min_amount = float(data["loans"]["loan_amount"].min())
max_amount = float(data["loans"]["loan_amount"].max())
amount_range = st.sidebar.slider(
    "Loan Amount Range",
    min_value=min_amount,
    max_value=max_amount,
    value=(min_amount, max_amount),
    format="$%0.0f"
)

# Apply filters
filtered_loans = data["loans"][
    (data["loans"]["time_period"] == selected_period) &
    (data["loans"]["industry"].isin(selected_industries)) &
    (data["loans"]["loan_purpose"].isin(selected_purposes)) &
    (data["loans"]["loan_amount"] >= amount_range[0]) &
    (data["loans"]["loan_amount"] <= amount_range[1])
]

# Portfolio overview metrics
st.header("Portfolio Overview")

# Calculate metrics
total_applications = len(filtered_loans)
total_loan_amount = filtered_loans["loan_amount"].sum()
average_interest_rate = filtered_loans["interest_rate"].mean()
approval_rate = filtered_loans[filtered_loans["approval_probability"] > 0.5].shape[0] / total_applications if total_applications > 0 else 0
average_risk_score = filtered_loans["risk_score"].mean()

# Display metrics
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Applications", f"{total_applications:,}")

with col2:
    st.metric("Total Loan Amount", f"${total_loan_amount:,.0f}")

with col3:
    st.metric("Avg. Interest Rate", f"{average_interest_rate*100:.2f}%")

with col4:
    st.metric("Approval Rate", f"{approval_rate:.1%}")

with col5:
    st.metric("Avg. Risk Score", f"{average_risk_score:.0f}")

# Portfolio distribution
st.header("Portfolio Distribution")

# Create tabs for different distribution views
tab1, tab2, tab3 = st.tabs(["By Industry", "By Risk Level", "By Loan Purpose"])

with tab1:
    # Distribution by industry
    industry_distribution = filtered_loans.groupby("industry").agg({
        "entity_id": "count",
        "loan_amount": "sum",
        "risk_score": "mean",
        "approval_probability": "mean"
    }).reset_index()
    
    industry_distribution.rename(columns={
        "entity_id": "application_count",
        "loan_amount": "total_amount",
        "risk_score": "avg_risk_score",
        "approval_probability": "avg_approval_probability"
    }, inplace=True)
    
    # Calculate percentages
    total_count = industry_distribution["application_count"].sum()
    total_amount = industry_distribution["total_amount"].sum()
    
    industry_distribution["count_percentage"] = industry_distribution["application_count"] / total_count
    industry_distribution["amount_percentage"] = industry_distribution["total_amount"] / total_amount
    
    # Sort by count
    industry_distribution = industry_distribution.sort_values("application_count", ascending=False)
    
    # Create visualizations
    fig1 = px.bar(
        industry_distribution,
        x="industry",
        y="application_count",
        title="Number of Applications by Industry",
        labels={"industry": "Industry", "application_count": "Number of Applications"},
        color="avg_risk_score",
        color_continuous_scale="RdYlGn_r"
    )
    
    st.plotly_chart(fig1, use_container_width=True)
    
    fig2 = px.pie(
        industry_distribution,
        values="total_amount",
        names="industry",
        title="Total Loan Amount by Industry",
        hover_data=["total_amount", "amount_percentage"]
    )
    
    fig2.update_traces(textposition='inside', textinfo='percent+label')
    
    st.plotly_chart(fig2, use_container_width=True)

with tab2:
    # Distribution by risk level
    # Create risk categories
    filtered_loans["risk_category"] = pd.cut(
        filtered_loans["risk_score"],
        bins=[0, 30, 60, 90, float('inf')],
        labels=["Low Risk", "Moderate Risk", "High Risk", "Very High Risk"]
    )
    
    risk_distribution = filtered_loans.groupby("risk_category").agg({
        "entity_id": "count",
        "loan_amount": "sum",
        "approval_probability": "mean"
    }).reset_index()
    
    risk_distribution.rename(columns={
        "entity_id": "application_count",
        "loan_amount": "total_amount",
        "approval_probability": "avg_approval_probability"
    }, inplace=True)
    
    # Calculate percentages
    risk_distribution["count_percentage"] = risk_distribution["application_count"] / risk_distribution["application_count"].sum()
    
    # Create visualizations
    fig3 = px.bar(
        risk_distribution,
        x="risk_category",
        y="application_count",
        title="Number of Applications by Risk Category",
        labels={"risk_category": "Risk Category", "application_count": "Number of Applications"},
        color="risk_category",
        color_discrete_map={
            "Low Risk": "green",
            "Moderate Risk": "yellow",
            "High Risk": "orange",
            "Very High Risk": "red"
        }
    )
    
    st.plotly_chart(fig3, use_container_width=True)
    
    # Create a gauge chart for portfolio risk distribution
    risk_gauge_data = [
        risk_distribution[risk_distribution["risk_category"] == "Low Risk"]["count_percentage"].sum(),
        risk_distribution[risk_distribution["risk_category"] == "Moderate Risk"]["count_percentage"].sum(),
        risk_distribution[risk_distribution["risk_category"] == "High Risk"]["count_percentage"].sum(),
        risk_distribution[risk_distribution["risk_category"] == "Very High Risk"]["count_percentage"].sum()
    ]
    
    # Account for any missing categories
    risk_gauge_data = [0 if pd.isna(x) else x for x in risk_gauge_data]
    
    fig4 = go.Figure()
    
    fig4.add_trace(go.Indicator(
        mode = "gauge+number",
        value = risk_gauge_data[0] + risk_gauge_data[1] * 0.5,
        title = {'text': "Portfolio Health"},
        gauge = {
            'axis': {'range': [0, 1], 'tickwidth': 1},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 0.33], 'color': "red"},
                {'range': [0.33, 0.67], 'color': "yellow"},
                {'range': [0.67, 1], 'color': "green"}
            ],
        }
    ))
    
    st.plotly_chart(fig4, use_container_width=True)
    
    st.markdown("""
    **Portfolio Health Gauge** shows a weighted measure of overall portfolio risk:
    - Higher values (green) indicate a healthier portfolio with more low-risk loans
    - Lower values (red) indicate higher overall portfolio risk
    """)

with tab3:
    # Distribution by loan purpose
    purpose_distribution = filtered_loans.groupby("loan_purpose").agg({
        "entity_id": "count",
        "loan_amount": "sum",
        "interest_rate": "mean",
        "risk_score": "mean"
    }).reset_index()
    
    purpose_distribution.rename(columns={
        "entity_id": "application_count",
        "loan_amount": "total_amount",
        "interest_rate": "avg_interest_rate",
        "risk_score": "avg_risk_score"
    }, inplace=True)
    
    # Sort by count
    purpose_distribution = purpose_distribution.sort_values("application_count", ascending=False)
    
    # Create visualizations
    fig5 = px.bar(
        purpose_distribution,
        x="loan_purpose",
        y="application_count",
        title="Number of Applications by Loan Purpose",
        labels={"loan_purpose": "Loan Purpose", "application_count": "Number of Applications"},
        color="avg_risk_score",
        color_continuous_scale="RdYlGn_r"
    )
    
    st.plotly_chart(fig5, use_container_width=True)
    
    # Create scatter plot of loan amount vs. interest rate by purpose
    fig6 = px.scatter(
        filtered_loans,
        x="loan_amount",
        y="interest_rate",
        color="loan_purpose",
        size="risk_score",
        hover_name="company_name",
        title="Loan Amount vs. Interest Rate by Purpose",
        labels={
            "loan_amount": "Loan Amount",
            "interest_rate": "Interest Rate",
            "loan_purpose": "Loan Purpose"
        }
    )
    
    # Update y-axis to show as percentage
    fig6.update_layout(yaxis_tickformat='.1%')
    
    st.plotly_chart(fig6, use_container_width=True)

# Loan application table
st.header("Loan Applications")

# Create a dataframe for display
display_loans = filtered_loans[
    ["company_name", "industry", "loan_amount", "interest_rate", "loan_purpose", 
     "term_years", "risk_score", "approval_status"]
].copy()

# Format values for display
display_loans["loan_amount"] = display_loans["loan_amount"].apply(lambda x: f"${x:,.0f}")
display_loans["interest_rate"] = display_loans["interest_rate"].apply(lambda x: f"{x*100:.2f}%")
display_loans["risk_score"] = display_loans["risk_score"].apply(lambda x: f"{x:.0f}")

# Show the table
st.dataframe(
    display_loans.sort_values("company_name"),
    use_container_width=True,
    hide_index=True
)

# Footer
st.markdown("---")
st.markdown("""
This portfolio view demonstrates how the knowledge graph can be used to analyze loan applications
at an aggregate level, providing insights into the overall risk profile and distribution of the portfolio.
""")