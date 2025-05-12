import streamlit as st
import pandas as pd

class RetailAppUI:
    def render_header(self):
        st.markdown("""
            <div style='text-align: center; margin-bottom: 2rem;'>
                <h1 style='margin-bottom: 0.2rem;'>🛒 Retail Sales AI Assistant</h1>
                <p style='font-size: 1.1rem;'>Ask questions about your retail data in plain English</p>
            </div>
        """, unsafe_allow_html=True)

    def render_query_section(self, initial_question):
        st.markdown("#### 💬 Type your question here:")
        question = st.text_area(
            label="Your Question",
            value=initial_question,
            placeholder="E.g. Who are our top 5 customers by total spending?",
            height=80,
            label_visibility="collapsed"
        )
        col1, col2 = st.columns([1, 1])
        execute_btn = col1.button("🔎 Get Answer", use_container_width=True)
        clear_btn = col2.button("🧹 Clear", use_container_width=True)
        return question, execute_btn, clear_btn

    def render_results(self, results):
        if not results:
            return

        if results.get("error"):
            st.error(f"❌ {results['error']}")
        else:
            st.success("✅ Query executed successfully")
            st.markdown("#### 🧾 Answer")
            st.code(results["query"], language="sql")
            st.markdown(results["formatted_text"])
            if isinstance(results["result_df"], pd.DataFrame) and not results["result_df"].empty:
                st.dataframe(results["result_df"])

    def render_chat_history(self, chat_history):
        with st.expander("🕓 Previous Questions", expanded=False):
            for entry in reversed(chat_history):
                st.markdown(f"**Q:** {entry['question']}")
                if entry["results"].get("error"):
                    st.error(f"Error: {entry['results']['error']}")
                else:
                    st.markdown(f"**Answer:** {entry['results']['formatted_text']}")
