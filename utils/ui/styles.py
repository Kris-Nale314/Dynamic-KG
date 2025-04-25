# utils/ui/styles.py

import streamlit as st

def load_css():
    """Load custom CSS to style the application with dark theme compatibility."""
    st.markdown("""
    <style>
    /* Dark theme compatible styling */
    :root {
        --primary-color: #7E57C2;
        --secondary-color: #B39DDB;
        --background-color: #121212;
        --surface-color: #1E1E1E;
        --text-color: #E0E0E0;
        --success-color: #81C784;
        --warning-color: #FFD54F;
        --error-color: #E57373;
    }
    
    /* Card styling with dark theme compatibility */
    .card {
        border-radius: 5px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        margin-bottom: 20px;
        background-color: var(--surface-color);
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
    }
    
    /* Badge styling */
    .badge {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 5px;
    }
    .badge-primary { background-color: var(--primary-color); color: white; }
    .badge-secondary { background-color: var(--secondary-color); color: var(--background-color); }
    .badge-success { background-color: var(--success-color); color: var(--background-color); }
    .badge-warning { background-color: var(--warning-color); color: var(--background-color); }
    .badge-error { background-color: var(--error-color); color: white; }
    
    /* Clean, minimal headings */
    h1, h2, h3 {
        font-weight: 400;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Subtle dividers */
    hr {
        border-color: rgba(255, 255, 255, 0.1);
    }
    
    /* Logo container */
    .logo-container {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        align-items: center;
        justify-content: center;
    }
    
    .logo-item {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        width: 80px;
        height: 90px;
        padding: 5px;
        border-radius: 5px;
    }
    
    .logo-item img {
        max-width: 60px;
        max-height: 60px;
        object-fit: contain;
    }
    
    .logo-item p {
        margin: 0;
        font-size: 0.7em;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

def set_page_config():
    """Set the standard page configuration for Dynamic-KG pages."""
    st.set_page_config(
        page_title="Dynamic-KG",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'About': "Dynamic-KG is an experimental framework for building evolving knowledge graphs."
        }
    )
    
    # Load custom CSS
    load_css()

def card(title, content, color="primary"):
    """Render a styled card compatible with dark theme."""
    st.markdown(f"""
    <div class="card">
        <h3 style="color: var(--{color}-color);">{title}</h3>
        <div>{content}</div>
    </div>
    """, unsafe_allow_html=True)

def badge(text, type="primary"):
    """Render a badge with the specified type."""
    return f'<span class="badge badge-{type}">{text}</span>'

def header_with_info(title, info_text):
    """Create a header with an info icon that shows a tooltip on hover."""
    st.markdown(f"### {title}", unsafe_allow_html=True)