import streamlit as st
import json
import os

# Users database file
USERS_FILE = "users.json"

def load_users():
    """Load users from JSON file"""
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    """Save users to JSON file"""
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f)

# Custom CSS for auth styling
AUTH_CSS = """
    <style>
    * {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Auth Container */
    .auth-container {
        max-width: 450px;
        margin: 80px auto;
        padding: 50px 40px;
        background: white;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        border-top: 4px solid #0066cc;
    }
    
    .auth-title {
        font-size: 1.8em;
        margin-bottom: 10px;
        color: #333;
        text-align: center;
        font-weight: 600;
    }
    
    .auth-subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 30px;
        font-size: 0.95em;
    }
    </style>
"""

def login_signup():
    """Display login/signup interface"""
    st.markdown(AUTH_CSS, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="auth-container">', unsafe_allow_html=True)
        st.markdown('<div class="auth-title">Course Material Recommender</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-subtitle">Sign up or log in to get started</div>', unsafe_allow_html=True)
        
        auth_mode = st.radio("Choose option", ["Login", "Sign Up"], horizontal=True, label_visibility="collapsed")
        
        st.write("")  # Spacing
        
        if auth_mode == "Sign Up":
            st.subheader("Create New Account")
            new_username = st.text_input("Username", key="signup_username", placeholder="Enter your username")
            new_password = st.text_input("Password", type="password", key="signup_password", placeholder="Enter your password")
            confirm_password = st.text_input("Confirm Password", type="password", key="signup_confirm", placeholder="Re-enter your password")
            new_email = st.text_input("Email (optional)", key="signup_email", placeholder="your.email@example.com")
            
            st.write("")  # Spacing
            
            if st.button("Sign Up", use_container_width=True):
                users = load_users()
                
                if not new_username or not new_password:
                    st.error("Username and password are required!")
                elif new_username in users:
                    st.error("Username already exists!")
                elif new_password != confirm_password:
                    st.error("Passwords don't match!")
                else:
                    users[new_username] = {
                        "password": new_password,
                        "email": new_email
                    }
                    save_users(users)
                    st.success("Account created successfully! Please log in.")
        
        else:  # Login
            st.subheader("Login to Your Account")
            username = st.text_input("Username", key="login_username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", key="login_password", placeholder="Enter your password")
            
            st.write("")  # Spacing
            
            if st.button("Login", use_container_width=True):
                users = load_users()
                
                if username in users and users[username]["password"] == password:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.success(f"Welcome back, {username}!")
                    st.rerun()
                elif username and password:
                    st.error("Invalid username or password!")
                else:
                    st.error("Please enter username and password!")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.stop()
