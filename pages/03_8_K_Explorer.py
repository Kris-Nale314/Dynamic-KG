"""
03_8k_Explorer.py

This page provides exploration and analysis of 8-K filings and material events,
showing timeline visualization, event clustering, and impact analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
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

# Import event processor
sys.path.append(str(Path(__file__).parents[1] / "core" / "processors"))
try:
    import event_processor as ep
except ImportError:
    st.error("Event processor module not found. Please make sure it exists in the core/processors directory.")
    event_processor_available = False
else:
    event_processor_available = True

# Initialize configuration
Config.initialize()

# Set up logger
logger = Config.get_logger("8k_explorer")
logger.info("8-K Explorer page loaded")

# Set page configuration
set_page_config()

# Track page view
SessionState.track_page_view("8K Explorer")

# Title and description
st.title("8-K Explorer")
st.markdown("""
Explore material events reported in 8-K filings across companies. This page visualizes event 
timelines, detects event clusters, and analyzes the impact of material events on company performance 
and industry trends.
""")

# Load data
@st.cache_data
def load_filing_data():
    """Load 8-K filing data for all companies."""
    if not event_processor_available:
        return None
    
    # Get all company directories
    company_dirs = [d.name for d in Config.COMPANIES_DIR.glob("*") if d.is_dir()]
    
    # Load filings for each company
    all_filings = []
    for ticker in company_dirs:
        filings = ep.load_8k_filings(ticker)
        if not filings.empty:
            all_filings.append(filings)
    
    # Combine all filings
    if all_filings:
        combined = pd.concat(all_filings)
        return combined
    else:
        return pd.DataFrame()

# Check if data directory exists
if not Config.COMPANIES_DIR.exists():
    st.error("Companies data directory not found. Please collect company data first.")
    st.stop()

# Load filing data
filings_data = load_filing_data()

if filings_data is None:
    st.error("Event processor module not available. Please make sure it's properly installed.")
    st.stop()

if filings_data.empty:
    st.warning("No 8-K filings data found. Please collect 8-K filing data first.")
    st.stop()

# Create tabs for different views
tab1, tab2, tab3, tab4 = st.tabs([
    "Event Timeline", "Event Clusters", "Impact Analysis", "Leading Indicators"
])

# Sidebar filters (applied to all tabs)
st.sidebar.header("Event Filters")

# Determine date column
date_column = None
for col in ['date', 'acceptedDate', 'filingDate']:
    if col in filings_data.columns:
        date_column = col
        filings_data[col] = pd.to_datetime(filings_data[col])
        break

if date_column is None:
    st.error("No date column found in filings data.")
    st.stop()

# Date range filter
min_date = filings_data[date_column].min().date()
max_date = filings_data[date_column].max().date()

selected_date_range = st.sidebar.date_input(
    "Date Range",
    value=(
        max_date - timedelta(days=180),
        max_date
    ),
    min_value=min_date,
    max_value=max_date
)

if len(selected_date_range) == 2:
    start_date, end_date = selected_date_range
else:
    start_date = selected_date_range[0]
    end_date = max_date

# Company filter
ticker_column = None
for col in ['ticker', 'symbol']:
    if col in filings_data.columns:
        ticker_column = col
        break

if ticker_column is None:
    st.error("No ticker column found in filings data.")
    st.stop()

# Get unique company tickers
companies = sorted(filings_data[ticker_column].unique())
selected_companies = st.sidebar.multiselect(
    "Select Companies",
    companies,
    default=companies[:5] if len(companies) > 5 else companies
)

# Create categorized event timeline
if event_processor_available:
    timeline = ep.create_event_timeline(
        filings_data,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        tickers=selected_companies if selected_companies else None
    )
else:
    timeline = filings_data

# Event Timeline Tab
with tab1:
    st.header("Event Timeline")
    
    if timeline.empty:
        st.info("No events found for the selected filters.")
    else:
        # Display timeline visualization
        st.subheader("Timeline Visualization")
        
        # Prepare data for visualization
        vis_data = timeline.copy()
        
        # Ensure we have business categories
        if 'business_categories' not in vis_data.columns:
            vis_data['business_categories'] = [['Other'] for _ in range(len(vis_data))]
        
        # Create a primary category column
        vis_data['primary_category'] = vis_data['business_categories'].apply(
            lambda x: x[0] if isinstance(x, list) and x else 'Other'
        )
        
        # Create visualization
        fig = px.scatter(
            vis_data,
            x=date_column,
            y=ticker_column,
            color='primary_category',
            hover_name='title' if 'title' in vis_data.columns else None,
            size_max=10,
            height=600
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Company",
            legend_title="Event Category"
        )
        
        # Add markers for different event types
        st.plotly_chart(fig, use_container_width=True)
        
        # Event list with details
        st.subheader("Event Details")
        
        # Allow filtering by event category
        all_categories = set()
        for cats in vis_data['business_categories']:
            if isinstance(cats, list):
                all_categories.update(cats)
        
        selected_categories = st.multiselect(
            "Filter by Category",
            sorted(all_categories),
            default=[]
        )
        
        # Apply category filter
        if selected_categories:
            filtered_events = vis_data[vis_data['business_categories'].apply(
                lambda x: any(cat in x for cat in selected_categories) if isinstance(x, list) else False
            )]
        else:
            filtered_events = vis_data
        
        # Sort by date
        filtered_events = filtered_events.sort_values(date_column, ascending=False)
        
        # Display events
        for _, event in filtered_events.iterrows():
            # Create card for each event
            event_date = event[date_column].strftime("%Y-%m-%d")
            event_company = event[ticker_column]
            event_title = event.get('title', 'Unnamed Event')
            
            # Get event categories
            event_cats = event.get('business_categories', [])
            cat_badges = ""
            if isinstance(event_cats, list):
                for cat in event_cats:
                    cat_badges += badge(cat, "primary") + " "
            
            # Get event items
            event_items = event.get('items', [])
            item_str = ""
            if isinstance(event_items, list):
                item_str = "<br>".join([f"• {item}: {ep.ITEM_CATEGORIES.get(item, '')}" for item in event_items])
            
            # Get link if available
            event_link = ""
            for link_col in ['finalLink', 'link', 'url']:
                if link_col in event and pd.notna(event[link_col]):
                    event_link = f"<a href='{event[link_col]}' target='_blank'>View Filing</a>"
                    break
            
            # Create expandable card
            with st.expander(f"{event_date} | {event_company} | {event_title}"):
                st.markdown(f"**Date:** {event_date}")
                st.markdown(f"**Company:** {event_company}")
                st.markdown(f"**Title:** {event_title}")
                st.markdown(f"**Categories:** {cat_badges}", unsafe_allow_html=True)
                
                if item_str:
                    st.markdown(f"**Items:**<br>{item_str}", unsafe_allow_html=True)
                
                if event_link:
                    st.markdown(f"{event_link}", unsafe_allow_html=True)

# Event Clusters Tab
with tab2:
    st.header("Event Clusters")
    
    if timeline.empty:
        st.info("No events found for the selected filters.")
    else:
        # Cluster settings
        st.subheader("Cluster Settings")
        col1, col2 = st.columns(2)
        
        with col1:
            date_window = st.slider(
                "Date Window (days)",
                min_value=1,
                max_value=30,
                value=7,
                help="Maximum number of days between events to be considered part of the same cluster"
            )
        
        with col2:
            min_cluster_size = st.slider(
                "Minimum Cluster Size",
                min_value=2,
                max_value=10,
                value=2,
                help="Minimum number of events required to form a cluster"
            )
        
        # Detect clusters
        if event_processor_available:
            clusters = ep.detect_event_clusters(
                timeline, 
                date_window=date_window,
                min_cluster_size=min_cluster_size
            )
        else:
            clusters = []
        
        if not clusters:
            st.info("No significant event clusters found with the current settings.")
        else:
            # Display clusters
            st.subheader(f"Found {len(clusters)} Event Clusters")
            
            # Create a summary table
            cluster_summary = []
            for i, cluster in enumerate(clusters):
                cluster_summary.append({
                    'Cluster': i + 1,
                    'Category': cluster['category'],
                    'Companies': len(cluster['companies']),
                    'Events': cluster['event_count'],
                    'Date Range': f"{cluster['start_date'].strftime('%Y-%m-%d')} to {cluster['end_date'].strftime('%Y-%m-%d')}",
                    'Duration': f"{cluster['date_range_days']} days"
                })
            
            summary_df = pd.DataFrame(cluster_summary)
            st.dataframe(summary_df, hide_index=True)
            
            # Show detail for each cluster
            for i, cluster in enumerate(clusters):
                with st.expander(f"Cluster {i+1}: {cluster['category']} ({cluster['event_count']} events)"):
                    st.markdown(f"**Category:** {cluster['category']}")
                    st.markdown(f"**Companies:** {', '.join(cluster['companies'])}")
                    st.markdown(f"**Date Range:** {cluster['start_date'].strftime('%Y-%m-%d')} to {cluster['end_date'].strftime('%Y-%m-%d')} ({cluster['date_range_days']} days)")
                    
                    # Create timeline for this cluster
                    events = pd.DataFrame(cluster['events'])
                    
                    # Create a timeline visualization
                    fig = px.scatter(
                        events,
                        x=date_column,
                        y=ticker_column,
                        hover_name='title' if 'title' in events.columns else None,
                        height=300
                    )
                    
                    # Update layout
                    fig.update_layout(
                        xaxis_title="Date",
                        yaxis_title="Company",
                        title="Cluster Events Timeline"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # List events in the cluster
                    st.markdown("**Events in this cluster:**")
                    for _, event in events.iterrows():
                        event_date = event[date_column].strftime("%Y-%m-%d")
                        event_company = event[ticker_column]
                        event_title = event.get('title', 'Unnamed Event')
                        
                        st.markdown(f"• {event_date} | {event_company} | {event_title}")

# Impact Analysis Tab
with tab3:
    st.header("Event Impact Analysis")
    
    # Select a company and event for impact analysis
    st.subheader("Select Event")
    
    # Company selector
    impact_company = st.selectbox(
        "Select Company for Impact Analysis",
        options=selected_companies if selected_companies else companies,
        index=0 if selected_companies or companies else None
    )
    
    if impact_company:
        # Filter timeline for this company
        company_events = timeline[timeline[ticker_column] == impact_company]
        
        if company_events.empty:
            st.info(f"No events found for {impact_company} in the selected date range.")
        else:
            # Create options for event selection
            event_options = []
            for _, event in company_events.iterrows():
                event_date = event[date_column].strftime("%Y-%m-%d")
                event_title = event.get('title', 'Event')
                event_options.append({
                    'label': f"{event_date} | {event_title}",
                    'date': event_date
                })
            
            # Sort by date
            event_options.sort(key=lambda x: x['date'], reverse=True)
            
            # Create display options
            event_labels = [opt['label'] for opt in event_options]
            
            if event_labels:
                selected_event_label = st.selectbox(
                    "Select Event",
                    options=event_labels,
                    index=0
                )
                
                # Get date for selected event
                selected_event_date = next(opt['date'] for opt in event_options if opt['label'] == selected_event_label)
                
                # Analyze impact
                if event_processor_available:
                    with st.spinner("Analyzing event impact..."):
                        # Set analysis window
                        window_days = st.slider(
                            "Analysis Window (days)",
                            min_value=1,
                            max_value=30,
                            value=7,
                            help="Number of days before and after the event to analyze"
                        )
                        
                        # Run impact analysis
                        impact = ep.analyze_event_impact(
                            impact_company,
                            selected_event_date,
                            window_days=window_days
                        )
                        
                        if impact['status'] == 'success':
                            # Display impact results
                            st.subheader("Stock Price Impact")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric(
                                    "Price Change",
                                    f"${impact['price_change']:.2f}",
                                    f"{impact['price_change_pct']:.2f}%"
                                )
                            
                            with col2:
                                st.metric(
                                    "Pre-Event Price",
                                    f"${impact['pre_event_price']:.2f}"
                                )
                            
                            with col3:
                                st.metric(
                                    "Post-Event Price",
                                    f"${impact['post_event_price']:.2f}"
                                )
                            
                            # Display price chart
                            price_data = pd.DataFrame(impact['price_series'])
                            event_date_dt = pd.to_datetime(selected_event_date)
                            
                            # Add event marker
                            fig = px.line(
                                price_data,
                                x='date',
                                y='close',
                                title=f"{impact_company} Stock Price Around Event"
                            )
                            
                            # Add vertical line for event date using shape
                            fig.add_shape(
                                type="line",
                                x0=event_date_dt,
                                x1=event_date_dt,
                                y0=0,
                                y1=1,
                                yref="paper",
                                line=dict(color="red", width=2, dash="dash")
                            )

                            # Add annotation for event date
                            fig.add_annotation(
                                x=event_date_dt,
                                y=1,
                                yref="paper",
                                text="Event Date",
                                showarrow=False,
                                font=dict(color="red")
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Volume analysis
                            st.subheader("Trading Volume Impact")
                            
                            vol_col1, vol_col2 = st.columns(2)
                            
                            with vol_col1:
                                st.metric(
                                    "Volume Change",
                                    f"{impact['volume_change']:,.0f}",
                                    f"{impact['volume_change_pct']:.2f}%"
                                )
                            
                            with vol_col2:
                                st.metric(
                                    "Avg. Pre-Event Volume",
                                    f"{impact['pre_event_volume']:,.0f}"
                                )
                            
                            # Display volume chart
                            fig = px.bar(
                                price_data,
                                x='date',
                                y='volume',
                                title=f"{impact_company} Trading Volume Around Event"
                            )
                            
                            # Add vertical line for event date using shape
                            fig.add_shape(
                                type="line",
                                x0=event_date_dt,
                                x1=event_date_dt,
                                y0=0,
                                y1=1,
                                yref="paper",
                                line=dict(color="red", width=2, dash="dash")
                            )

                            # Add annotation for event date
                            fig.add_annotation(
                                x=event_date_dt,
                                y=1,
                                yref="paper",
                                text="Event Date",
                                showarrow=False,
                                font=dict(color="red")
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Peer analysis
                            st.subheader("Peer Company Reactions")
                            
                            # Get potential peers
                            if impact_company in selected_companies:
                                potential_peers = [c for c in selected_companies if c != impact_company]
                            else:
                                potential_peers = companies[:5] if len(companies) > 5 else companies
                                potential_peers = [p for p in potential_peers if p != impact_company]
                            
                            # Let user select peers
                            selected_peers = st.multiselect(
                                "Select Peer Companies",
                                options=[c for c in companies if c != impact_company],
                                default=potential_peers[:3] if len(potential_peers) > 3 else potential_peers
                            )
                            
                            if selected_peers:
                                with st.spinner("Analyzing peer reactions..."):
                                    peer_analysis = ep.analyze_peer_reactions(
                                        impact_company,
                                        selected_event_date,
                                        peers=selected_peers,
                                        window_days=window_days
                                    )
                                    
                                    if peer_analysis['status'] == 'success' and peer_analysis['peer_impacts']:
                                        # Display correlation
                                        if peer_analysis['correlations']:
                                            st.subheader("Price Movement Correlation")
                                            
                                            # Create correlation chart
                                            corr_data = []
                                            for peer, corr in peer_analysis['correlations'].items():
                                                corr_data.append({
                                                    'Peer': peer,
                                                    'Correlation': corr
                                                })
                                            
                                            corr_df = pd.DataFrame(corr_data)
                                            
                                            # Add color based on correlation strength
                                            fig = px.bar(
                                                corr_df,
                                                x='Peer',
                                                y='Correlation',
                                                color='Correlation',
                                                color_continuous_scale='RdBu_r',
                                                range_color=[-1, 1],
                                                title="Price Movement Correlation with Event Company"
                                            )
                                            
                                            st.plotly_chart(fig, use_container_width=True)
                                        
                                        # Display peer price changes
                                        st.subheader("Peer Company Price Changes")
                                        
                                        peer_impact_data = []
                                        for peer_impact in peer_analysis['peer_impacts']:
                                            peer_impact_data.append({
                                                'Company': peer_impact['ticker'],
                                                'Price Change (%)': peer_impact['price_change_pct'],
                                                'Price Change ($)': peer_impact['price_change'],
                                                'Volume Change (%)': peer_impact['volume_change_pct']
                                            })
                                        
                                        # Add the main company for comparison
                                        peer_impact_data.append({
                                            'Company': impact_company,
                                            'Price Change (%)': impact['price_change_pct'],
                                            'Price Change ($)': impact['price_change'],
                                            'Volume Change (%)': impact['volume_change_pct']
                                        })
                                        
                                        impact_df = pd.DataFrame(peer_impact_data)
                                        
                                        # Create chart
                                        fig = px.bar(
                                            impact_df,
                                            x='Company',
                                            y='Price Change (%)',
                                            color='Price Change (%)',
                                            color_continuous_scale='RdBu',
                                            title="Price Changes Across Companies"
                                        )
                                        
                                        st.plotly_chart(fig, use_container_width=True)
                                    else:
                                        st.info("Not enough data available for peer analysis.")
                            else:
                                st.info("Please select peer companies for comparison.")
                        else:
                            st.warning(f"Impact analysis failed: {impact.get('message', 'Unknown error')}")
                else:
                    st.warning("Event processor not available. Cannot perform impact analysis.")
            else:
                st.info("No events available for the selected company and date range.")
    else:
        st.info("Please select a company for impact analysis.")

# Leading Indicators Tab
with tab4:
    st.header("Leading Indicators Analysis")
    st.info("""
    This feature analyzes patterns in 8-K filings to identify potential leading indicators 
    for company performance changes. It requires financial performance data over time.
    """)
    
    # Check if financial data is available
    st.subheader("Select Analysis Parameters")
    
    # Company selector
    indicator_companies = st.multiselect(
        "Select Companies to Analyze",
        options=companies,
        default=companies[:3] if len(companies) > 3 else companies
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        lag_days = st.slider(
            "Maximum Lag Period (days)",
            min_value=30,
            max_value=365,
            value=90,
            step=30,
            help="Maximum number of days to look for relationships between events and performance changes"
        )
    
    with col2:
        min_occurrences = st.slider(
            "Minimum Occurrences",
            min_value=2,
            max_value=10,
            value=3,
            help="Minimum number of occurrences required to consider a pattern significant"
        )
    
    # Create a placeholder for future implementation
    st.info("""
    This analysis requires financial performance data over time for the selected companies.
    
    In a future version, this tab will:
    
    1. Identify event types that tend to precede performance changes
    2. Calculate the average lag time between events and impacts
    3. Measure the consistency of these relationships
    4. Suggest potential leading indicators for risk assessment
    """)
    
    # Placeholder for visualization
    st.markdown("""
    ### Sample Event-Impact Relationship
    
    The chart below is a conceptual example of how event types might correlate with performance changes:
    """)
    
    # Create a sample visualization
    sample_data = pd.DataFrame({
        'Event_Type': ['Leadership Changes', 'Financial Restatements', 'Material Agreements', 
                       'Asset Dispositions', 'Debt Restructuring'],
        'Avg_Impact': [0.15, -0.23, 0.08, -0.05, 0.12],
        'Consistency': [0.85, 0.78, 0.65, 0.58, 0.73],
        'Occurrences': [8, 5, 12, 7, 4]
    })
    
    fig = px.scatter(
        sample_data,
        x='Consistency',
        y='Avg_Impact',
        size='Occurrences',
        color='Avg_Impact',
        hover_name='Event_Type',
        color_continuous_scale='RdBu',
        size_max=30,
        title="Event Impact Analysis (Example)"
    )
    
    fig.update_layout(
        xaxis_title="Pattern Consistency",
        yaxis_title="Average Performance Impact",
        xaxis=dict(tickformat='.0%'),
        yaxis=dict(tickformat='.0%')
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
This 8-K Explorer demonstrates how the Dynamic-KG can track material events over time and analyze
their impact on companies and industries. The temporal dimension of the knowledge graph allows for
tracking how events and their effects evolve and propagate.
""")