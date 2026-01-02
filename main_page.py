import streamlit as st
from main import recommend_materials
import uuid

# Custom CSS for main page styling
MAIN_PAGE_CSS = """
    <style>
    * {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Main Header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 40px 30px;
        border-radius: 12px;
        margin-bottom: 40px;
        color: white;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.2em;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    .main-header p {
        margin: 10px 0 0 0;
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.05em;
    }
    
    /* Search Section */
    .search-section {
        background: white;
        padding: 30px;
        border-radius: 12px;
        border: 1px solid #e8e8e8;
        margin: 20px 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        transition: box-shadow 0.3s ease;
    }
    
    .search-section:hover {
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
    }
    
    .search-section h2 {
        margin-top: 0;
        color: #222;
        font-size: 1.4em;
        font-weight: 600;
    }
    
    /* Results Container */
    .result-container {
        background: linear-gradient(to bottom, #f8f9fa, #ffffff);
        padding: 30px;
        border-radius: 12px;
        margin-top: 30px;
        border: 1px solid #e8e8e8;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    .result-container h2 {
        margin-top: 0;
        color: #222;
        font-size: 1.4em;
        font-weight: 600;
    }
    
    .material-card {
        background: linear-gradient(135deg, #ffffff 0%, #f5f7fb 100%);
        padding: 20px;
        margin: 15px 0;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        transition: all 0.3s ease;
    }
    
    .material-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        border-left-color: #764ba2;
    }
    
    /* Sidebar Styling */
    .sidebar-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
        font-weight: 600;
        text-align: center;
    }
    
    .sidebar-section {
        margin-bottom: 25px;
    }
    
    .sidebar-section h3 {
        color: #222;
        font-size: 1em;
        font-weight: 600;
        margin-bottom: 10px;
    }
    
    /* Example buttons */
    .example-btn {
        background: #f5f7fb;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 10px;
        margin: 8px 0;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .example-btn:hover {
        background: #667eea;
        color: white;
        border-color: #667eea;
    }
    
    /* Footer */
    .footer {
        border-top: 1px solid #e0e0e0;
        padding: 30px 0;
        margin-top: 60px;
        color: #666;
        font-size: 0.9em;
        text-align: center;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #667eea;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 500;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #764ba2;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    /* Input Fields */
    .stTextInput > div > div > input,
    .stPasswordInput > div > div > input {
        border-radius: 8px;
        border: 1px solid #ddd;
        padding: 12px 14px;
        font-size: 1em;
    }
    
    .stTextInput > div > div > input:focus,
    .stPasswordInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Info/Warning/Error Messages */
    .stInfo {
        background-color: #f0f4ff;
        border-radius: 8px;
        padding: 15px 20px;
        border-left: 4px solid #667eea;
    }
    
    .stWarning {
        background-color: #fffbf0;
        border-radius: 8px;
        padding: 15px 20px;
        border-left: 4px solid #ffc107;
    }
    
    .stError {
        background-color: #fff5f5;
        border-radius: 8px;
        padding: 15px 20px;
        border-left: 4px solid #e74c3c;
    }
    
    .stSuccess {
        background-color: #f0fff4;
        border-radius: 8px;
        padding: 15px 20px;
        border-left: 4px solid #27ae60;
    }
    
    /* Stats Cards */
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        margin: 10px;
        height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        box-sizing: border-box;
    }
    
    .stat-card h3 {
        margin: 0;
        font-size: 2em;
        font-weight: 700;
    }
    
    .stat-card p {
        margin: 10px 0 0 0;
        color: rgba(255, 255, 255, 0.9);
    }
    </style>
"""

def show_main_page():
    """Display the main chat interface"""
    # Apply CSS styling
    st.markdown(MAIN_PAGE_CSS, unsafe_allow_html=True)
    
    # Header section
    st.markdown("""
        <div class="main-header">
            <h1>Course Material Recommender</h1>
            <p>Discover the perfect study materials for every course</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for instructions and examples
    with st.sidebar:
        st.markdown("""
            <div class="sidebar-header">
                Welcome, {0}!
            </div>
        """.format(st.session_state.username), unsafe_allow_html=True)
        
        if st.button("Logout", use_container_width=True, key="logout_btn"):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.rerun()
        
        st.divider()
        
        st.markdown("### How It Works")
        st.info("**Step 1:** Enter course code or material type\n\n**Step 2:** Click Search\n\n**Step 3:** Get instant results")
        
        st.markdown("### Quick Search")
        examples = [
            "CSE 2101 book",
            "slides on programming",
            "CSE 2104 lab manual",
            "CSE 3105 notes",
            "CSE 1101 video"
        ]
        for example in examples:
            if st.button(example, use_container_width=True, key=example):
                st.session_state.query = example
    
    # Main query section - Chat Interface
    st.markdown("### Chat with Material Recommender")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Chat input
    user_input = st.chat_input("Ask me anything about course materials... (e.g., 'CSE 2101 book' or 'slides for programming')")
    
    # Statistics Section - Browse by Category
    st.markdown("### Browse by Category")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
            <div class="stat-card">
                <h3>Books</h3>
                <p>Textbooks & Books</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="stat-card">
                <h3>Slides</h3>
                <p>Slides & Presentations</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="stat-card">
                <h3 style="font-size: 1.5em;">Lab Manuals</h3>
                <p style="font-size: 0.9em;">Lab Manuals</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
            <div class="stat-card">
                <h3>Notes</h3>
                <p>Notes & Handouts</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Display chat messages with stable pair IDs so Delete removes the correct user+assistant
    messages = st.session_state.get("messages", [])
    i = 0
    while i < len(messages):
        msg = messages[i]
        role = msg.get("role", "assistant")
        pair_id = msg.get("pair_id", str(uuid.uuid4()))  # fallback: generate if missing
    
        if role == "user":
            # show user message with Delete button inside the same bubble
            with st.chat_message("user"):
                st.markdown(msg.get("content", ""))
                if st.button("Delete", key=f"delete_{pair_id}"):
                    # remove all messages with this pair_id (user and assistant)
                    st.session_state.messages = [m for m in st.session_state.messages if m.get("pair_id") != pair_id]
                    break
    
            # find corresponding assistant message (if any) and display it, then skip it
            assistant_index = None
            for j in range(i + 1, len(messages)):
                if messages[j].get("pair_id") == pair_id and messages[j].get("role") == "assistant":
                    assistant_index = j
                    break
    
            if assistant_index is not None:
                with st.chat_message("assistant"):
                    st.markdown(messages[assistant_index].get("content", ""))
                i = assistant_index + 1
            else:
                i += 1
    
        else:
            # display non-user messages (fallback)
            with st.chat_message(role):
                st.markdown(msg.get("content", ""))
            i += 1
    
    if user_input:
        # Generate a unique pair_id for this user/assistant pair
        pair_id = str(uuid.uuid4())
        
        # Add user message to chat history with pair_id
        st.session_state.messages.append({"role": "user", "content": user_input, "pair_id": pair_id})
        
        # Display user message immediately
        st.chat_message("user").markdown(user_input)
        
        # Get response from recommendation system
        with st.spinner("Searching through materials..."):
            response = recommend_materials(user_input)
        
        # Add assistant response to chat history with same pair_id
        st.session_state.messages.append({"role": "assistant", "content": response, "pair_id": pair_id})
        
        # Display assistant response immediately
        st.chat_message("assistant").markdown(response)
    
    # Sidebar: Clear chat button (placed after main UI so it renders in sidebar earlier section exists)
    with st.sidebar:
        if st.button("Clear Chat", key="clear_chat_btn"):
            st.session_state.messages = []
    
    st.markdown("---")
