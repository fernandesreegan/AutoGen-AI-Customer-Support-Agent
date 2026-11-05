import streamlit as st
import requests
import pandas as pd
import threading
import uvicorn
from backend.main import app as fastapi_app

# ==========================================
# 1. AUTO-START FASTAPI BACKEND IN BACKGROUND
# ==========================================
@st.cache_resource
def start_backend():
    def run_server():
        uvicorn.run(fastapi_app, host="127.0.0.1", port=8000, log_level="warning")
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

start_backend()
API_URL = "http://127.0.0.1:8000"

# ==========================================
# 2. STREAMLIT PAGE CONFIG & CUSTOM CSS
# ==========================================
st.set_page_config(
    page_title="AI Customer Support Hub",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 3.2rem !important;
        color: #1E3A8A;
        font-weight: 800;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4B5563;
        margin-bottom: 30px;
    }
    div.stButton > button {
        background-color: #2563EB !important;
        color: white !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        border: none !important;
        transition: background-color 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: #1D4ED8 !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 3. SIDEBAR NAVIGATION & INFO
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/customer-support.png", width=80)
    st.title("Support Hub Info")
    st.markdown("---")
    st.markdown("### 🧠 Autonomous Agent Team")
    st.info(
        "1. **Query Agent**: Analyzes intent & sentiment.\n"
        "2. **Knowledge Agent**: FAISS vector RAG search[cite: 5].\n"
        "3. **Resolution Agent**: Generates tailored answers[cite: 1].\n"
        "4. **Log Agent**: Evaluates escalation rules & saves records[cite: 3]."
    )
    st.markdown("---")
    st.markdown("**Status**: 🟢 System Online")
    st.markdown("**Version**: 2.1.2 Enterprise")

# ==========================================
# 4. MAIN INTERFACE & STATE INITIALIZATION
# ==========================================
st.markdown('<p class="main-header">🤖 AI Customer Support Agent Automation</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Seamlessly triage, resolve, and log customer complaints using a 4-agent AutoGen team.</p>', unsafe_allow_html=True)

if "agent_history" not in st.session_state:
    st.session_state.agent_history = []
if "success_msg" not in st.session_state:
    st.session_state.success_msg = ""
if "form_version" not in st.session_state:
    st.session_state.form_version = 0
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False

col1, col2 = st.columns([1.5, 1], gap="large")

with col1:
    st.markdown("### 📝 Submit New Ticket")
    
    # Standard inputs using version keys to allow programmatic resetting
    customer_name = st.text_input("👤 Customer Name", value="Fernandes Reegan", key=f"name_{st.session_state.form_version}")
    order_id = st.text_input("📦 Order ID", value="ORD-4512", key=f"order_{st.session_state.form_version}")
    complaint = st.text_area("💬 Customer Complaint", height=130, placeholder="Type detailed customer issue here...", key=f"complaint_{st.session_state.form_version}")
    
    # The button is explicitly locked when is_processing is True
    submit_btn = st.button(
        "🚀 Process Complaint through Agents", 
        type="primary", 
        use_container_width=True,
        disabled=st.session_state.is_processing
    )

with col2:
    st.markdown("### 📊 Live Analytics")
    try:
        metrics_res = requests.get(f"{API_URL}/logs", timeout=2)
        if metrics_res.status_code == 200:
            m_data = metrics_res.json().get("logs", [])
            total_tickets = len(m_data)
            escalated_count = sum(1 for x in m_data if str(x.get("status", "")).lower() == "escalated")
            resolved_count = total_tickets - escalated_count
            
            mc1, mc2 = st.columns(2)
            with mc1:
                st.metric(label="Total Tickets", value=total_tickets)
            with mc2:
                st.metric(label="Escalated", value=escalated_count, delta_color="inverse")
            st.metric(label="Resolved Successfully", value=resolved_count)
        else:
            st.info("Analytics loading...")
    except Exception:
        st.info("Start submitting tickets to populate metrics.")

# ==========================================
# 5. AGENT EXECUTION HANDLER (WITH PRECISE LOCKING)
# ==========================================
if submit_btn:
    current_complaint = complaint.strip()
    if not current_complaint:
        st.warning("⚠️ Please enter a complaint before submitting.")
    else:
        # Lock the button immediately
        st.session_state.is_processing = True
        st.session_state.success_msg = ""
        st.rerun()

# If processing flag is active, execute the backend request securely
if st.session_state.is_processing and complaint.strip():
    current_complaint = complaint.strip()
    with st.spinner("⚙️ AutoGen agents collaborating (Query ➔ Knowledge ➔ Resolution ➔ Log)..."):
        try:
            res = requests.post(
                f"{API_URL}/process-ticket",
                json={"customer_name": customer_name, "order_id": order_id, "complaint": current_complaint}
            )
            if res.status_code == 200:
                data = res.json()
                st.session_state.agent_history = data.get("agent_history", [])
                st.session_state.success_msg = "🎉 Ticket Processed & Logged Successfully!"
                
                # Unlock button, increment version to clear text box cleanly, and rerun
                st.session_state.is_processing = False
                st.session_state.form_version += 1
                st.rerun()
            else:
                st.error(f"❌ Server Error: {res.text}")
                st.session_state.is_processing = False
        except Exception as e:
            st.error(f"❌ Connection failed: {e}")
            st.session_state.is_processing = False

# Display Success Notification if present
if st.session_state.success_msg:
    st.success(st.session_state.success_msg)

# Display Persistent Agent Conversation Flow if available
if st.session_state.agent_history:
    st.markdown("### 🔄 Agent Conversation & Action Flow")
    for step in st.session_state.agent_history:
        with st.expander(f"📌 Agent: {step['agent']}", expanded=True):
            st.write(step['content'])

# ==========================================
# 6. TICKET LOGS & EXPORT SECTION
# ==========================================
st.markdown("---")
st.subheader("📂 Central Ticket Logs Table")

if st.button("🔄 Refresh Logs"):
    pass

try:
    log_res = requests.get(f"{API_URL}/logs")
    if log_res.status_code == 200:
        logs_data = log_res.json().get("logs", [])
        if logs_data:
            df = pd.DataFrame(logs_data)
            st.dataframe(df, use_container_width=True)
            
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Complete Logs (CSV)",
                data=csv_data,
                file_name="support_tickets_log.csv",
                mime="text/csv"
            )
        else:
            st.info("ℹ️ No tickets logged yet. Submit a complaint using the form above.")
    else:
        st.warning("⚠️ Waiting for backend service response...")
except Exception:
    st.info("ℹ️ Backend starting up... click 'Refresh Logs' in a moment.")