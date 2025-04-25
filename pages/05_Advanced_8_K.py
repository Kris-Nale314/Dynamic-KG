"""
04_Event_Analytics.py

Advanced event analytics for 8-K filings and market data. This page focuses on exploring
the relationships between material events and financial metrics to identify potential
leading and lagging indicators.
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
from typing import Dict, List, Any, Optional

# Add the project root directory to the Python path for importing config
root_dir = str(Path(__file__).parents[1].absolute())
if root_dir not in sys.path:
    sys.path.append(root_dir)

# Import project modules
from config import Config
from utils.ui.styles import set_page_config, card, badge
from utils.ui.session import SessionState

# Import event processor and analytics modules
sys.path.append(str(Path(__file__).parents[1] / "core" / "processors"))
import event_processor as ep
import event_analytics as ea

# Helper function for event category colors
def get_color_for_category(category):
    """Return a consistent color for an event category."""
    category_colors = {
        "Leadership Changes": "#FF6B6B",  # Red
        "Financial Condition": "#4ECDC4",  # Teal
        "Material Agreements": "#FFD166",  # Yellow
        "Corporate Actions": "#6B5B95",    # Purple
        "Asset Transactions": "#88D8B0",   # Green
        "Accounting Issues": "#FF8C42",    # Orange
        "Financial Restructuring": "#F25F5C",  # Coral
        "Securities Matters": "#59A5D8",   # Blue
        "Other Disclosures": "#9A8C98"     # Gray
    }
    return category_colors.get(category, "#BBBBBB")  # Default to gray

# Initialize configuration
Config.initialize()

# Set up logger
logger = Config.get_logger("event_analytics_page")
logger.info("Event Analytics page loaded")

# Set page configuration
set_page_config()

# Track page view
SessionState.track_page_view("Event Analytics")

# Title and description
st.title("Event Analytics")
st.markdown("""
Explore relationships between material events (8-K filings) and financial metrics. 
This page helps identify potential leading and lagging indicators by analyzing how 
different types of events correlate with market changes over time.
""")

# Define target companies
TARGET_COMPANIES = [
    # Technology
    "GOOGL", "MSFT", "DELL", "NVDA", "TSM",
    # Consumer
    "TSLA", "AMZN", "WMT", "MCD", "NKE",
    # Industrial/Energy
    "SHEL", "CAT", "BA", "GE", "UNP", "CEG",
    # Financial
    "JPM", "GS", "V", "AXP", "BRK-A",
    # Healthcare
    "JNJ", "UNH", "PFE",
    # Professional Services
    "BAH", "ACN",
    # Real Estate
    "PLD", "SPG"
]

# Define key metrics categories
KEY_METRICS = {
    "Price Movement": [
        "1-day price change",
        "5-day price change", 
        "30-day price change"
    ],
    "Trading Activity": [
        "30-day volatility",
        "trading volume change"
    ],
    "Market Context": [
        "s&p 500 change",
        "10y treasury yield change",
        "yield curve change"
    ]
}

# Helper functions for data handling
def prepare_event_data(events_df):
    """Prepare event data by handling list columns properly."""
    df_copy = events_df.copy()
    
    # Convert list columns to strings for processing
    for col in df_copy.columns:
        if df_copy[col].apply(lambda x: isinstance(x, list)).any():
            df_copy[col] = df_copy[col].apply(lambda x: json.dumps(x) if isinstance(x, list) else x)
    
    return df_copy

def restore_event_data(events_df):
    """Restore list columns from strings."""
    df_copy = events_df.copy()
    
    for col in ['business_categories', 'items']:
        if col in df_copy.columns:
            try:
                df_copy[col] = df_copy[col].apply(
                    lambda x: json.loads(x) if isinstance(x, str) and x.startswith('[') else x
                )
            except:
                pass  # Skip if conversion fails
    
    return df_copy

# Load event data
@st.cache_data
def load_event_data():
    """Load 8-K filing data and create timeline."""
    try:
        # Load filings using event processor
        filings = ep.load_8k_filings()
        
        if filings is not None and not filings.empty:
            # Prepare and create timeline
            prepared_filings = prepare_event_data(filings)
            timeline = ep.create_event_timeline(prepared_filings)
            
            # Restore list columns
            timeline = restore_event_data(timeline)
            
            return timeline
        else:
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading filing data: {e}")
        return pd.DataFrame()

# Load the data
with st.spinner("Loading 8-K filing data..."):
    event_timeline = load_event_data()

# Check if we have data
if event_timeline.empty:
    st.error("""
    No 8-K filing data available. Please run the data collection script first:
    ```
    python data/external/collect_fmp_data.py
    ```
    """)
    st.stop()

# Find necessary columns
date_col = None
for col in ['date', 'acceptedDate', 'filingDate']:
    if col in event_timeline.columns:
        date_col = col
        event_timeline[col] = pd.to_datetime(event_timeline[col])
        break

ticker_col = None
for col in ['ticker', 'symbol']:
    if col in event_timeline.columns:
        ticker_col = col
        break

if not date_col or not ticker_col:
    st.error("Required date or ticker columns not found in event data.")
    st.stop()

# Sidebar controls
st.sidebar.header("Analysis Controls")

# Company selection
available_companies = event_timeline[ticker_col].unique()
target_companies_available = sorted([c for c in TARGET_COMPANIES if c in available_companies])

if not target_companies_available:
    st.sidebar.warning("No target companies found in data.")
    company_list = sorted(available_companies)
else:
    company_list = target_companies_available

selection_mode = st.sidebar.radio(
    "Company Selection",
    ["All Target Companies", "Select Companies"],
    index=0
)

if selection_mode == "All Target Companies":
    selected_companies = target_companies_available
    st.sidebar.caption(f"Analyzing all {len(selected_companies)} target companies")
else:
    selected_companies = st.sidebar.multiselect(
        "Select Companies",
        options=company_list,
        default=company_list[:5] if len(company_list) > 5 else company_list
    )
    
    if not selected_companies:
        st.sidebar.warning("Please select at least one company")
        selected_companies = company_list[:1]

# Event type selection
event_categories = set()
if 'business_categories' in event_timeline.columns:
    for cats in event_timeline['business_categories']:
        if isinstance(cats, list):
            event_categories.update(cats)

selected_categories = st.sidebar.multiselect(
    "Event Types",
    options=sorted(event_categories),
    default=sorted(event_categories)[:3] if len(event_categories) >= 3 else sorted(event_categories)
)

# Analysis timeframe
date_range = [event_timeline[date_col].min().date(), event_timeline[date_col].max().date()]
st.sidebar.caption(f"Data available from {date_range[0]} to {date_range[1]}")

# Window size for analysis
window_days = st.sidebar.slider(
    "Analysis Window (days)",
    min_value=5,
    max_value=90,
    value=30,
    help="Number of days after an event to analyze impact"
)

# Analysis button
run_analysis = st.sidebar.button(
    "Run Analysis",
    type="primary"
)

# Filter the data based on selections
def filter_events(df, companies, categories):
    """Filter events based on user selections."""
    # Filter by company
    if companies:
        filtered = df[df[ticker_col].isin(companies)]
    else:
        filtered = df.copy()
    
    # Filter by categories if selected
    if categories:
        def has_category(cats):
            if isinstance(cats, list):
                return any(cat in categories for cat in cats)
            return False
        
        filtered = filtered[filtered['business_categories'].apply(has_category)]
    
    return filtered

# Apply filters
filtered_events = filter_events(
    event_timeline,
    selected_companies,
    selected_categories
)

event_count = len(filtered_events)
st.sidebar.metric("Events Found", f"{event_count}")

# Create main interface with tabs
tab1, tab2, tab3 = st.tabs([
    "Event Timeline", 
    "Event-Metric Relationships", 
    "Company Interconnections"
])

# Tab 1: Event Timeline
with tab1:
    st.header("Event Timeline")
    
    if event_count == 0:
        st.info("No events match your selection criteria.")
    else:
        # Sort events by date
        timeline_events = filtered_events.sort_values(date_col)
        
        # Create time density chart
        st.subheader("Event Density Over Time")
        
        # Group by month and category for density chart
        timeline_events['month'] = timeline_events[date_col].dt.to_period('M')
        
        # Count events by month and category
        monthly_counts = []
        
        for _, row in timeline_events.iterrows():
            month = row['month']
            if isinstance(row['business_categories'], list):
                for category in row['business_categories']:
                    monthly_counts.append({
                        'month': month,
                        'category': category,
                        'count': 1
                    })
        
        # Convert to DataFrame and aggregate
        if monthly_counts:
            density_df = pd.DataFrame(monthly_counts)
            density_df = density_df.groupby(['month', 'category']).sum().reset_index()
            density_df['month'] = density_df['month'].astype(str)
            
            # Create stacked bar chart
            fig = px.bar(
                density_df,
                x='month',
                y='count',
                color='category',
                title="Monthly Event Distribution by Category",
                labels={'month': 'Month', 'count': 'Number of Events', 'category': 'Event Type'},
                color_discrete_map={cat: get_color_for_category(cat) for cat in density_df['category'].unique()},
            )
            
            fig.update_layout(
                xaxis_title="Month",
                yaxis_title="Event Count",
                legend_title="Event Type",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("About This Chart"):
                st.markdown("""
                This chart shows the distribution of different event types over time. 
                Look for patterns such as:
                
                - Seasonal clustering of certain event types
                - Increasing or decreasing trends in specific categories
                - Periods with unusual event activity
                
                These patterns can help identify potential leading indicators or
                early warning signals for market changes.
                """)
        
        # Create detailed event timeline
        st.subheader("Detailed Event Timeline")
        
        # Create figure
        fig = go.Figure()
        
        # Add companies as y-axis
        companies_in_timeline = sorted(timeline_events[ticker_col].unique())
        
        # Create dictionary mapping company to row position
        company_positions = {company: i for i, company in enumerate(companies_in_timeline)}
        
        # Add horizontal lines for companies
        for company in companies_in_timeline:
            fig.add_shape(
                type="line",
                x0=timeline_events[date_col].min(),
                x1=timeline_events[date_col].max(),
                y0=company_positions[company],
                y1=company_positions[company],
                line=dict(color="lightgray", width=1)
            )
        
        # Add events as markers
        for _, event in timeline_events.iterrows():
            company = event[ticker_col]
            event_date = event[date_col]
            
            # Get category for color
            category = "Other"
            if isinstance(event['business_categories'], list) and event['business_categories']:
                category = event['business_categories'][0]
            
            # Get title for hover text
            title = f"{company} Event"
            if 'title' in event and event['title']:
                title = event['title']
            
            # Add marker
            fig.add_trace(go.Scatter(
                x=[event_date],
                y=[company_positions[company]],
                mode="markers",
                marker=dict(
                    size=10,
                    color=get_color_for_category(category),
                    line=dict(width=1, color="white")
                ),
                name=category,
                text=f"{event_date.strftime('%Y-%m-%d')}: {title}",
                hoverinfo="text",
                showlegend=False
            ))
        
        # Update layout
        fig.update_layout(
            title="Event Timeline by Company",
            xaxis_title="Date",
            yaxis=dict(
                tickmode="array",
                tickvals=list(company_positions.values()),
                ticktext=list(company_positions.keys()),
                title="Company"
            ),
            height=max(400, len(companies_in_timeline) * 30),
            margin=dict(l=140, r=20, t=60, b=60),
            hovermode="closest"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("About This Chart"):
            st.markdown("""
            This timeline shows individual events by company over time. Each dot represents 
            an 8-K filing, with color indicating the event type. Hover over dots to see details.
            
            Look for:
            
            - Clustering of events across companies
            - Sequential patterns (events in one company followed by similar events in others)
            - Correlation between event timing and market movements
            """)
        
        # Summary statistics
        st.subheader("Event Summary")
        
        # Get summary stats
        if 'business_categories' in timeline_events.columns:
            # Count events by category
            category_counts = {}
            for cats in timeline_events['business_categories']:
                if isinstance(cats, list):
                    for cat in cats:
                        category_counts[cat] = category_counts.get(cat, 0) + 1
            
            # Display as horizontal bar chart
            if category_counts:
                category_df = pd.DataFrame({
                    'Category': list(category_counts.keys()),
                    'Count': list(category_counts.values())
                }).sort_values('Count', ascending=False)
                
                fig = px.bar(
                    category_df,
                    y='Category',
                    x='Count',
                    orientation='h',
                    color='Category',
                    color_discrete_map={cat: get_color_for_category(cat) for cat in category_df['Category']},
                    title="Event Count by Category"
                )
                
                fig.update_layout(
                    showlegend=False,
                    height=max(300, len(category_counts) * 30),
                    yaxis=dict(categoryorder='total ascending')
                )
                
                st.plotly_chart(fig, use_container_width=True)

# Tab 2: Event-Metric Relationships
with tab2:
    st.header("Event-Metric Relationships")
    
    if event_count < 5:
        st.info("Not enough events for meaningful analysis. Select more companies or event types.")
    else:
        # Run or load analysis
        relationship_results = None
        
        if run_analysis:
            with st.spinner("Analyzing event-metric relationships..."):
                try:
                    # Prepare filtered events
                    filtered_events_prepared = restore_event_data(filtered_events)
                    
                    # Run analysis using event_analytics module
                    relationship_results = ea.analyze_event_metric_relationships(
                        filtered_events_prepared,
                        window_days=window_days
                    )
                except Exception as e:
                    st.error(f"Error analyzing relationships: {str(e)}")
                    logger.error(f"Error in relationship analysis: {e}", exc_info=True)
        
        if relationship_results:
            # Create subtabs for different visualizations
            rel_tab1, rel_tab2 = st.tabs(["Correlation Matrix", "Impact Analysis"])
            
            with rel_tab1:
                st.subheader("Event-Metric Correlation")
                
                if 'correlation_matrix' in relationship_results and 'event_types' in relationship_results and 'metric_names' in relationship_results:
                    # Extract data
                    corr_matrix = np.array(relationship_results['correlation_matrix'])
                    event_types = relationship_results['event_types']
                    metric_names = relationship_results['metric_names']
                    
                    if len(event_types) > 0 and len(metric_names) > 0 and corr_matrix.size > 0:
                        # Create heatmap
                        fig = px.imshow(
                            corr_matrix,
                            x=metric_names,
                            y=event_types,
                            color_continuous_scale="RdBu_r",
                            labels=dict(color="Correlation"),
                            zmin=-1, zmax=1
                        )
                        
                        fig.update_layout(
                            xaxis_title="Metrics",
                            yaxis_title="Event Types",
                            height=500,
                            xaxis=dict(tickangle=45)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add explanation
                        st.markdown("""
                        This matrix reveals how different types of events correlate with various financial metrics:
                        
                        - **Blue cells**: Events positively correlated with metrics (increases)
                        - **Red cells**: Events negatively correlated with metrics (decreases)
                        - **Darker colors**: Stronger relationships
                        
                        This helps identify which events could serve as leading indicators for specific market movements.
                        """)
                    else:
                        st.info("Not enough data for correlation analysis.")
                else:
                    st.info("Correlation data not available in analysis results.")
            
            with rel_tab2:
                st.subheader("Event Impact Analysis")
                
                if 'avg_changes' in relationship_results and 'event_types' in relationship_results:
                    # Get event types with data
                    event_options = [et for et in relationship_results['event_types'] 
                                    if et in relationship_results['avg_changes']]
                    
                    if event_options:
                        # Create event selector
                        selected_event = st.selectbox(
                            "Select Event Type",
                            options=event_options
                        )
                        
                        # Display impact for selected event
                        if selected_event in relationship_results['avg_changes']:
                            event_changes = relationship_results['avg_changes'][selected_event]
                            
                            if event_changes:
                                # Create DataFrame for visualization
                                changes_data = []
                                
                                for metric, value in event_changes.items():
                                    # Find category for this metric
                                    category = "Other"
                                    for cat, metrics in KEY_METRICS.items():
                                        if metric in metrics:
                                            category = cat
                                            break
                                    
                                    changes_data.append({
                                        "Metric": metric,
                                        "Category": category,
                                        "Average Change": value
                                    })
                                
                                changes_df = pd.DataFrame(changes_data)
                                
                                if not changes_df.empty:
                                    # Sort by category and change magnitude
                                    changes_df['abs_change'] = changes_df['Average Change'].abs()
                                    changes_df = changes_df.sort_values(['Category', 'abs_change'], ascending=[True, False])
                                    
                                    # Create visualization
                                    fig = px.bar(
                                        changes_df,
                                        x="Metric",
                                        y="Average Change",
                                        color="Category",
                                        title=f"Average Impact of {selected_event} Events"
                                    )
                                    
                                    # Update layout
                                    fig.update_layout(
                                        xaxis_tickangle=45,
                                        yaxis=dict(
                                            tickformat=".1%",
                                            title="Average Change"
                                        )
                                    )
                                    
                                    # Add zero line
                                    fig.add_hline(
                                        y=0, 
                                        line_dash="dash", 
                                        line_color="gray"
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Add explanation
                                    st.markdown(f"""
                                    This chart shows the average impact of **{selected_event}** events on different metrics:
                                    
                                    - **Bars above zero**: Metrics that tend to increase
                                    - **Bars below zero**: Metrics that tend to decrease
                                    - **Bar height**: Magnitude of change
                                    
                                    This reveals the typical market reaction to this type of event, helping identify
                                    potential leading indicators for risk assessment.
                                    """)
                            else:
                                st.info(f"No impact data available for {selected_event} events.")
                        else:
                            st.info(f"No data available for {selected_event} events.")
                    else:
                        st.info("No event impact data available in analysis results.")
                else:
                    st.info("Event impact data not available in analysis results.")
                
                # Add lag analysis if available
                if 'lag_analysis' in relationship_results and relationship_results['lag_analysis']:
                    st.subheader("Timing of Event Impacts")
                    
                    # Convert to DataFrame
                    lag_df = pd.DataFrame(relationship_results['lag_analysis'])
                    
                    if not lag_df.empty:
                        # Allow filtering by event type
                        event_types = sorted(lag_df['event_type'].unique())
                        
                        if event_types:
                            # Create scatter plot
                            fig = px.scatter(
                                lag_df,
                                x="avg_days",
                                y="correlation",
                                color="event_type",
                                size="occurrence_count",
                                hover_name="metric",
                                title="When Do Event Impacts Peak?",
                                labels={
                                    "avg_days": "Days After Event",
                                    "correlation": "Impact Strength",
                                    "event_type": "Event Type",
                                    "occurrence_count": "Frequency"
                                }
                            )
                            
                            # Update layout
                            fig.update_layout(
                                xaxis=dict(
                                    title="Days After Event",
                                    tickmode="linear",
                                    dtick=5
                                ),
                                yaxis=dict(
                                    title="Impact Strength",
                                    range=[-1, 1]
                                ),
                                height=500
                            )
                            
                            # Add zero line
                            fig.add_hline(y=0, line_dash="dash", line_color="gray")
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Add explanation
                            st.markdown("""
                            This chart shows when the impacts of different events typically peak:
                            
                            - **X-axis**: Days after event occurrence
                            - **Y-axis**: Strength of impact
                            - **Point size**: How frequently this pattern appears
                            
                            This helps determine the lead/lag time of different indicators and how
                            quickly markets typically react to different types of events.
                            """)
                        else:
                            st.info("No event types found in lag analysis.")
                    else:
                        st.info("No lag analysis data available.")
        else:
            st.info("Click 'Run Analysis' to analyze event-metric relationships.")

# Tab 3: Company Interconnections
with tab3:
    st.header("Company Interconnections")
    
    if event_count < 5:
        st.info("Not enough events for meaningful analysis. Select more companies or event types.")
    else:
        # Run or load propagation analysis
        propagation_results = None
        
        if run_analysis:
            with st.spinner("Analyzing company interconnections..."):
                try:
                    # Prepare filtered events
                    filtered_events_prepared = restore_event_data(filtered_events)
                    
                    # Run analysis using event_analytics module
                    propagation_results = ea.analyze_signal_propagation(
                        filtered_events_prepared,
                        window_days=window_days
                    )
                except Exception as e:
                    st.error(f"Error analyzing propagation: {str(e)}")
                    logger.error(f"Error in propagation analysis: {e}", exc_info=True)
        
        if propagation_results:
            # Display network visualization
            st.subheader("Company Relationship Network")
            
            if 'network_html' in propagation_results and propagation_results['network_html']:
                try:
                    # Display the network visualization
                    st.components.v1.html(
                        propagation_results['network_html'],
                        height=600
                    )
                    
                    # Add explanation
                    st.markdown("""
                    This network shows how events and their impacts propagate between companies:
                    
                    - **Nodes**: Companies in the dataset
                    - **Edges**: Relationships between companies
                    - **Edge color**: Green for positive correlations, red for negative
                    - **Edge thickness**: Strength of relationship
                    
                    This visualization reveals which companies tend to lead or follow in terms of
                    event impacts and how closely different companies are connected in the market.
                    """)
                except Exception as e:
                    st.error(f"Error displaying network visualization: {str(e)}")
            else:
                # Generate network on the fly
                with st.spinner("Generating company network..."):
                    try:
                        # Use event processor to build network
                        network = ep.build_event_correlation_network(
                            filtered_events,
                            min_correlation=0.2,
                            time_window=90
                        )
                        
                        if network and network.get('nodes') and network.get('edges'):
                            # Display network using NetworkX and Plotly
                            import networkx as nx
                            
                            # Create NetworkX graph
                            G = nx.Graph()
                            
                            # Add nodes
                            for node in network['nodes']:
                                G.add_node(
                                    node['id'],
                                    categories=node.get('categories', []),
                                    event_count=node.get('event_count', 0)
                                )
                            
                            # Add edges
                            for edge in network['edges']:
                                G.add_edge(
                                    edge['source'],
                                    edge['target'],
                                    weight=edge['correlation']
                                )
                            
                            # Create positions
                            pos = nx.spring_layout(G, seed=42)
                            
                            # Create edge traces
                            edge_traces = []
                            
                            # Add edges
                            for edge in network['edges']:
                                source = edge['source']
                                target = edge['target']
                                x0, y0 = pos[source]
                                x1, y1 = pos[target]
                                weight = abs(edge['correlation']) * 3  # Scale thickness
                                color = "green" if edge['correlation'] > 0 else "red"
                                
                                edge_trace = go.Scatter(
                                    x=[x0, x1, None],
                                    y=[y0, y1, None],
                                    line=dict(width=weight, color=color),
                                    hoverinfo='text',
                                    text=f"{source} to {target}: {edge['correlation']:.2f}",
                                    mode='lines'
                                )
                                
                                edge_traces.append(edge_trace)
                            
                            # Create node trace
                            node_trace = go.Scatter(
                                x=[pos[node][0] for node in G.nodes()],
                                y=[pos[node][1] for node in G.nodes()],
                                mode='markers',
                                hoverinfo='text',
                                marker=dict(
                                    size=[10 + G.nodes[node].get('event_count', 0) / 2 for node in G.nodes()],
                                    color=[get_node_color(G.nodes[node].get('categories', [])) for node in G.nodes()],
                                    line=dict(width=1, color='white')
                                ),
                                text=[f"{node}: {G.nodes[node].get('event_count', 0)} events" for node in G.nodes()]
                            )
                            
                            # Create figure
                            fig = go.Figure(data=edge_traces + [node_trace])
                            
                            # Update layout
                            fig.update_layout(
                                title="Company Relationship Network",
                                showlegend=False,
                                hovermode='closest',
                                margin=dict(b=20, l=5, r=5, t=40),
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                height=600
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Add explanation
                            st.markdown("""
                            This network shows relationships between companies based on event patterns:
                            
                            - **Nodes**: Companies in the dataset
                            - **Node size**: Number of events
                            - **Edges**: Correlations between event patterns
                            - **Edge color**: Green for positive correlations, red for negative
                            - **Edge thickness**: Strength of correlation
                            
                            Companies connected by thicker lines have more closely related event patterns.
                            """)
                        else:
                            st.info("Not enough relationship data to build a network visualization.")
                    except Exception as e:
                        st.error(f"Error generating network: {str(e)}")
                        logger.error(f"Network visualization error: {e}", exc_info=True)
            
            # Propagation timeline if available
            if 'timeline' in propagation_results and propagation_results['timeline']:
                st.subheader("Signal Propagation Timeline")
                
                # Convert to DataFrame
                timeline_df = pd.DataFrame(propagation_results['timeline'])
                
                if not timeline_df.empty:
                    # Event type selector
                    event_types = sorted(timeline_df['event_type'].unique())
                    
                    if event_types:
                        selected_event_type = st.selectbox(
                            "Select Event Type to Visualize",
                            options=event_types
                        )
                        
                        # Filter for selected event
                        event_timeline = timeline_df[timeline_df['event_type'] == selected_event_type]
                        
                        if not event_timeline.empty:
                            # Create visualization
                            fig = px.scatter(
                                event_timeline,
                                x="days_after_event",
                                y="impact_strength",
                                color="company",
                                size="significance",
                                hover_data=["source_company"],
                                title=f"Signal Propagation for {selected_event_type} Events",
                                labels={
                                    "days_after_event": "Days After Event",
                                    "impact_strength": "Impact Strength",
                                    "company": "Affected Company",
                                    "significance": "Significance",
                                    "source_company": "Source Company"
                                }
                            )
                            
                            # Update layout
                            fig.update_layout(
                                xaxis=dict(
                                    title="Days After Event",
                                    tickmode="linear", 
                                    dtick=5
                                ),
                                yaxis=dict(
                                    title="Impact Strength",
                                    tickformat=".1%"
                                ),
                                height=500
                            )
                            
                            # Add reference line at y=0
                            fig.add_hline(y=0, line_dash="dash", line_color="gray")
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Add explanation
                            st.markdown(f"""
                            This chart shows how the impact of **{selected_event_type}** events propagates across companies:
                            
                            - **X-axis**: Days after the event
                            - **Y-axis**: Strength of impact (positive or negative)
                            - **Point size**: Significance of the impact
                            - **Color**: Company experiencing the impact
                            
                            This visualization reveals the lag time between events at one company and 
                            their impacts on other companies, helping identify which companies are
                            leaders vs. followers in market movements.
                            """)
                        else:
                            st.info(f"No propagation data available for {selected_event_type} events.")
                    else:
                        st.info("No event types found in propagation timeline.")
                else:
                    st.info("No propagation timeline data available.")
            
            # Industry impact if available
            if 'industry_impact' in propagation_results and propagation_results['industry_impact']:
                st.subheader("Industry-Wide Impact")
                
                # Convert to DataFrame
                impact_df = pd.DataFrame(propagation_results['industry_impact'])
                
                if not impact_df.empty:
                    # Create visualization
                    fig = px.bar(
                        impact_df,
                        x="industry",
                        y="avg_impact",
                        color="event_type",
                        barmode="group",
                        title="Industry Impact by Event Type",
                        labels={
                            "industry": "Industry",
                            "avg_impact": "Average Impact",
                            "event_type": "Event Type"
                        }
                    )
                    
                    # Update layout
                    fig.update_layout(
                        xaxis_title="Industry",
                        yaxis_title="Average Impact",
                        yaxis=dict(tickformat=".1%"),
                        height=500
                    )
                    
                    # Add reference line at y=0
                    fig.add_hline(y=0, line_dash="dash", line_color="gray")
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add explanation
                    st.markdown("""
                    This chart shows how different event types impact entire industries:
                    
                    - **X-axis**: Industry sectors
                    - **Y-axis**: Average impact (positive or negative)
                    - **Bar groups**: Different types of events
                    
                    This helps identify which industries are most sensitive to specific
                    event types and which events have the broadest market impact.
                    """)
                else:
                    st.info("No industry impact data available.")
        else:
            st.info("Click 'Run Analysis' to analyze company interconnections.")

# Helper function for node colors
def get_node_color(categories):
    """Get a color for a node based on its most common category."""
    if not categories:
        return "#BBBBBB"
    
    # Count categories
    category_counts = {}
    for cat in categories:
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    # Get most common category
    if category_counts:
        most_common = max(category_counts.items(), key=lambda x: x[1])[0]
        return get_color_for_category(most_common)
    
    return "#BBBBBB"

# Footer
st.markdown("---")
st.markdown("""
**About This Analysis**

This page explores 8-K filings and their relationship to market metrics to help identify 
potential leading and lagging indicators for risk assessment. The analysis forms the basis 
for the Dynamic-KG project, which aims to create evolving knowledge graphs that capture 
temporal relationships between entities and events.

Data is sourced from SEC filings and market data for target companies across various sectors.
""")