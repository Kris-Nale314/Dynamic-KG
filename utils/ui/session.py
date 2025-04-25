# utils/ui/session.py

import streamlit as st
import uuid
from datetime import datetime

class SessionState:
    """Manage Streamlit session state variables."""
    
    @staticmethod
    def get_session_id():
        """Get or create a unique session ID."""
        if "session_id" not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
        return st.session_state.session_id
    
    @staticmethod
    def track_page_view(page_name):
        """Track page view analytics."""
        if "page_views" not in st.session_state:
            st.session_state.page_views = {}
        
        current_time = datetime.now().isoformat()
        
        if page_name not in st.session_state.page_views:
            st.session_state.page_views[page_name] = []
        
        st.session_state.page_views[page_name].append(current_time)
    
    @staticmethod
    def save_user_selection(key, value):
        """Save user selection for persistence between page reloads."""
        if "user_selections" not in st.session_state:
            st.session_state.user_selections = {}
        
        st.session_state.user_selections[key] = value
    
    @staticmethod
    def get_user_selection(key, default=None):
        """Get a previously saved user selection."""
        if "user_selections" not in st.session_state:
            return default
        
        return st.session_state.user_selections.get(key, default)
    
    @staticmethod
    def clear_selections():
        """Clear all saved user selections."""
        if "user_selections" in st.session_state:
            st.session_state.user_selections = {}