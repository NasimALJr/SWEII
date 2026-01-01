import streamlit as st
from login import login_signup
from main_page import show_main_page

# Page configuration
st.set_page_config(
    page_title="Course Material Recommender",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None

# Route based on login state
if not st.session_state.logged_in:
    login_signup()
else:
    show_main_page()

