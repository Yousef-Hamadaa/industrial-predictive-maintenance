import os
import sys
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.advisor.rag_advisor import AIMaintenanceAdvisor

st.set_page_config(page_title="AI Maintenance Advisor", page_icon="🤖", layout="wide")

st.title("🤖 AI Maintenance Advisor & Grounded Knowledge Assistant")
st.markdown("Ask questions about machine failure modes (TWF, HDF, PWF, OSF, RNF), physical sensor thresholds, or step-by-step maintenance protocols.")

advisor = AIMaintenanceAdvisor()

user_query = st.text_input("Enter your question or issue (e.g. 'How to handle HDF failure?'):", value="How to handle HDF failure?")

if user_query:
    response = advisor.answer_query(user_query)

    st.subheader(f"📌 {response['title']}")
    st.write(response['description'])

    st.markdown("#### 🚨 Key Operating Indicators:")
    for ind in response['indicators']:
        st.markdown(f"- `{ind}`")

    st.markdown("#### 🛠️ Step-by-Step Action Plan:")
    for step in response['action_plan']:
        st.write(step)
