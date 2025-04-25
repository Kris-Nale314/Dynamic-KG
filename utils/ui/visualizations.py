# utils/ui/visualizations.py

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def time_series_chart(df, date_column, value_columns, title="Time Series", height=400):
    """Create a time series chart with multiple series."""
    fig = go.Figure()
    
    for column in value_columns:
        fig.add_trace(
            go.Scatter(
                x=df[date_column],
                y=df[column],
                name=column,
                mode='lines'
            )
        )
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return st.plotly_chart(fig, use_container_width=True)

def comparison_bar_chart(df, category_column, value_column, comparison_column, title="Comparison", height=400):
    """Create a grouped bar chart for comparisons."""
    fig = px.bar(
        df,
        x=category_column,
        y=value_column,
        color=comparison_column,
        barmode='group',
        title=title,
        height=height
    )
    
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return st.plotly_chart(fig, use_container_width=True)

def create_gauge_chart(value, min_val=0, max_val=1, title="", threshold_ranges=None):
    """Create a gauge chart for displaying a value within a range."""
    if threshold_ranges is None:
        threshold_ranges = [
            {'min': 0, 'max': 0.33, 'color': 'red'},
            {'min': 0.33, 'max': 0.67, 'color': 'yellow'},
            {'min': 0.67, 'max': 1, 'color': 'green'}
        ]
    
    # Create the steps for the gauge
    steps = []
    for range_info in threshold_ranges:
        step = {
            'range': [range_info['min'], range_info['max']],
            'color': range_info['color']
        }
        steps.append(step)
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        title = {'text': title},
        gauge = {
            'axis': {'range': [min_val, max_val]},
            'steps': steps,
            'bar': {'color': "darkblue"}
        }
    ))
    
    return st.plotly_chart(fig, use_container_width=True)

def relationship_network(nodes_df, edges_df, node_id_col, source_col, target_col, 
                        node_color_col=None, node_size_col=None, edge_weight_col=None, 
                        title="Relationship Network"):
    """Create a network visualization of relationships."""
    # Ensure networkx is available
    try:
        import networkx as nx
    except ImportError:
        st.error("networkx package is required for network visualizations.")
        return None
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes
    for _, row in nodes_df.iterrows():
        node_id = row[node_id_col]
        node_attrs = {col: row[col] for col in nodes_df.columns if col != node_id_col}
        G.add_node(node_id, **node_attrs)
    
    # Add edges
    for _, row in edges_df.iterrows():
        source = row[source_col]
        target = row[target_col]
        edge_attrs = {col: row[col] for col in edges_df.columns 
                     if col not in [source_col, target_col]}
        G.add_edge(source, target, **edge_attrs)
    
    # Create PyVis network
    try:
        from pyvis.network import Network
        import tempfile
        
        # Create network
        net = Network(notebook=False, height="600px", width="100%", bgcolor="#ffffff", 
                    font_color="black")
        
        # Set options
        net.set_options("""
        {
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -50,
              "centralGravity": 0.01,
              "springLength": 100,
              "springConstant": 0.08
            },
            "maxVelocity": 50,
            "solver": "forceAtlas2Based",
            "timestep": 0.35,
            "stabilization": {
              "enabled": true,
              "iterations": 1000
            }
          }
        }
        """)
        
        # Calculate node colors and sizes if specified
        if node_color_col is not None:
            node_colors = nodes_df[node_color_col]
            color_map = {val: f"#{hash(str(val)) % 0xffffff:06x}" for val in node_colors.unique()}
        
        if node_size_col is not None:
            # Normalize sizes between 10 and 50
            sizes = nodes_df[node_size_col]
            min_size, max_size = sizes.min(), sizes.max()
            size_range = max_size - min_size
            norm_sizes = {row[node_id_col]: 10 + ((row[node_size_col] - min_size) / size_range * 40) 
                         if size_range > 0 else 25 
                         for _, row in nodes_df.iterrows()}
        
        # Add nodes to network
        for node_id in G.nodes():
            node_data = G.nodes[node_id]
            node_title = "<br>".join([f"{k}: {v}" for k, v in node_data.items()])
            
            # Set node color
            if node_color_col is not None and node_color_col in node_data:
                color = color_map.get(node_data[node_color_col], "#7CAEFF")
            else:
                color = "#7CAEFF"  # Default color
            
            # Set node size
            if node_size_col is not None:
                size = norm_sizes.get(node_id, 25)
            else:
                size = 25
            
            net.add_node(node_id, title=node_title, color=color, size=size, label=str(node_id))
        
        # Add edges to network
        for source, target, data in G.edges(data=True):
            weight = data.get(edge_weight_col, 1) if edge_weight_col is not None else 1
            edge_title = "<br>".join([f"{k}: {v}" for k, v in data.items()])
            net.add_edge(source, target, title=edge_title, value=weight)
        
        # Generate HTML file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp:
            net.save_graph(tmp.name)
            html_file = tmp.name
        
        # Display in Streamlit
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        st.components.v1.html(html_content, height=600)
        
        # Clean up
        import os
        os.unlink(html_file)
        
    except ImportError:
        st.error("pyvis package is required for network visualizations.")
        
        # Fallback to matplotlib
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='skyblue', 
                node_size=1500, edge_color='gray')
        st.pyplot(plt)

def create_correlation_heatmap(df, title="Correlation Matrix"):
    """Create a correlation heatmap for numerical columns."""
    # Calculate correlation matrix
    numeric_df = df.select_dtypes(include=['number'])
    corr = numeric_df.corr()
    
    # Create heatmap
    fig = px.imshow(
        corr,
        x=corr.columns,
        y=corr.columns,
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1,
        title=title
    )
    
    fig.update_layout(height=600)
    
    return st.plotly_chart(fig, use_container_width=True)

# Add these functions to your utils/ui/visualizations.py file

def create_histogram(df, column, title="Histogram", x_title=None, y_title=None, height=400):
    """Create a histogram for a column in a DataFrame."""
    fig = px.histogram(
        df,
        x=column,
        title=title,
        labels={column: x_title or column},
        height=height
    )
    
    fig.update_layout(
        yaxis_title=y_title or "Count"
    )
    
    return st.plotly_chart(fig, use_container_width=True)

def create_donut_chart(data_dict, title="Donut Chart", height=400):
    """Create a donut chart from a dictionary of values."""
    labels = list(data_dict.keys())
    values = list(data_dict.values())
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=.4
    )])
    
    fig.update_layout(
        title=title,
        height=height
    )
    
    return st.plotly_chart(fig, use_container_width=True)

def create_stacked_bar(df, x, y, color, title="Stacked Bar Chart", height=400):
    """Create a stacked bar chart from a DataFrame."""
    fig = px.bar(
        df,
        x=x,
        y=y,
        color=color,
        title=title,
        height=height,
        barmode="stack"
    )
    
    return st.plotly_chart(fig, use_container_width=True)