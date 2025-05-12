# === app2.py + Refined UI (matches mockup) ===
import streamlit as st
from retail_model import IntegratedRetailQueryModel
from retail_ui import RetailAppUI
import os
from dotenv import load_dotenv

load_dotenv()

def init_session_defaults():
    defaults = {
        "use_llm": True,
        "question": "",
        "results": None,
        "chat_history": []
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

def render_sidebar_info(model, db_name):
    with st.sidebar:
        st.markdown("### 🛍️ About This App")
        st.write("""
        This app lets you query your retail sales database using natural language.
        Just type your question, and our AI will convert it to SQL and fetch the results.
        """)
        st.markdown("---")
        st.markdown("### 🗄️ Database Connection")
        st.success("✅ Connected to database")
        st.markdown(f"**Database:** `{db_name}`")

        with st.expander("Available Tables"):
            for table in model.db.get_usable_table_names():
                st.write(f"- {table}")

def handle_query_execution(model, question, method_map, selected_method):
    st.session_state.question = question
    with st.spinner("🔄 Finding the answer to your question..."):
        results = model.execute_query(question, force_method=method_map[selected_method])
        st.info(f"Query processed using: {results.get('method', 'Unknown')}")
        if not results.get("error"):
            st.session_state.chat_history.append({"question": question, "results": results})
        st.session_state.results = results

def render_interactive_examples(examples):
    st.markdown("""
    <h4 style='margin-top: 2rem;'>Try one of these examples:</h4>
    """, unsafe_allow_html=True)
    cols = st.columns(2)
    for idx, example in enumerate(examples):
        with cols[idx % 2]:
            if st.button(f"📄 {example}"):
                st.session_state.question = example
                st.rerun()

def main():
    ui = RetailAppUI()
    ui.render_header()
    init_session_defaults()

    db_settings = {
        "user": os.getenv("DB_USER", "root"),
        "password": os.getenv("DB_PASSWORD", "root"),
        "host": os.getenv("DB_HOST", "localhost"),
        "name": os.getenv("DB_NAME", "retail_sales_db")
    }

    try:
        model = IntegratedRetailQueryModel(
            db_user=db_settings["user"],
            db_password=db_settings["password"],
            db_host=db_settings["host"],
            db_name=db_settings["name"],
            use_llm=st.session_state.use_llm
        )
        render_sidebar_info(model, db_settings["name"])
    except Exception as e:
        st.error(f"Failed to connect: {e}")
        with st.expander("Connection troubleshooting"):
            st.write("""
            - Ensure MySQL server is running
            - Verify credentials and DB name
            - Try a local DB client to test
            """)
        st.stop()

    question, execute_btn, clear_btn = ui.render_query_section(st.session_state.question)

    examples = [
        "Which product category had the highest sales last month?",
        "Who are our top 5 customers by total spending?",
        "Show me the average purchase amount by customer age group",
        "Compare sales performance between online and in-store channels",
        "What's the monthly trend of cosmetics sales over the past year?",
        "Identify products with inventory levels below safety stock",
        "Calculate the profit margin for each product category",
        "What day of the week has the highest foot traffic in our stores?",
        "Show me the total revenue from the electronics department"
    ]
    render_interactive_examples(examples)

    if clear_btn:
        st.session_state.question = ""
        st.session_state.results = None
        st.rerun()

    selected_method = st.radio("", ["Auto", "Force LLM", "Force Rule-based"], horizontal=True)
    method_map = {"Auto": None, "Force LLM": "llm", "Force Rule-based": "rules"}

    if execute_btn and question.strip():
        handle_query_execution(model, question, method_map, selected_method)

    ui.render_results(st.session_state.results)
    if st.session_state.chat_history:
        ui.render_chat_history(st.session_state.chat_history)

if __name__ == "__main__":
    main()
