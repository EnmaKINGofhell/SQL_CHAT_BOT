import os
import streamlit as st
import pandas as pd
import numpy as np
from langchain.chains import create_sql_query_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from sqlalchemy import create_engine
from sqlalchemy.exc import ProgrammingError
from langchain_community.utilities import SQLDatabase
from dotenv import load_dotenv
from decimal import Decimal

# Load environment variables
load_dotenv()

# ========== MODEL LOGIC ==========
class RetailQueryModel:
    def __init__(self, db_user, db_password, db_host, db_name):
        self.db_user = db_user
        self.db_password = db_password
        self.db_host = db_host
        self.db_name = db_name
        self.db = self._connect_to_database()
        self.llm = self._initialize_llm()
    
    def _connect_to_database(self):
        """Create database connection"""
        try:
            # Create SQLAlchemy engine
            connection_string = f"mysql+pymysql://{self.db_user}:{self.db_password}@{self.db_host}/{self.db_name}"
            engine = create_engine(connection_string)
            
            # Initialize SQLDatabase
            return SQLDatabase(engine, sample_rows_in_table_info=3)
        except Exception as e:
            raise ConnectionError(f"Database connection failed: {str(e)}")
    
    def _initialize_llm(self):
        """Initialize language model"""
        try:
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment variables")
                
            return ChatGoogleGenerativeAI(
                model="models/gemini-1.5-pro",
                temperature=0.2,
                google_api_key=api_key
            )
        except Exception as e:
            raise RuntimeError(f"LLM initialization failed: {str(e)}")
    
    def extract_sql_query(self, response):
        """Extract SQL query from LLM response"""
        # Case 1: Inside ```sql ... ```
        if "```sql" in response:
            return response.split("```sql")[1].split("```")[0].strip()
        
        # Case 2: Contains "SQLQuery:" prefix
        elif "SQLQuery:" in response:
            return response.split("SQLQuery:")[1].strip().split("\n")[0]
        
        # Case 3: First SELECT/UPDATE/etc. line fallback
        else:
            lines = response.splitlines()
            for line in lines:
                if line.strip().upper().startswith(("SELECT", "UPDATE", "DELETE", "INSERT")):
                    return line.strip()
        
        # If we can't parse it, return the whole response
        return response
    
    def parse_result(self, raw_result):
        """Convert raw SQL result to a DataFrame"""
        # If already a DataFrame
        if isinstance(raw_result, pd.DataFrame):
            return raw_result
        
        # Handle string representation of tuple results (the problematic case)
        if isinstance(raw_result, str):
            # First, check if it's the specific pattern we're seeing with age groups
            if raw_result.startswith("[") and "Decimal" in raw_result:
                try:
                    # Try to safely evaluate the string as a Python expression
                    import ast
                    tuples_list = ast.literal_eval(raw_result)
                    
                    # Check if it's a list of tuples with 2 elements (like age and amount)
                    if isinstance(tuples_list, list) and all(isinstance(item, tuple) and len(item) == 2 for item in tuples_list):
                        # Convert Decimal objects to float for better display
                        processed_data = [(item[0], float(item[1])) for item in tuples_list]
                        # Create DataFrame with appropriate column names
                        if all(isinstance(item[0], int) for item in processed_data):
                            return pd.DataFrame(processed_data, columns=["Age", "AveragePurchaseAmount"])
                        else:
                            return pd.DataFrame(processed_data, columns=["Category", "Value"])
                except (ValueError, SyntaxError):
                    # If evaluation fails, continue to other parsing methods
                    pass
        
        # Handle list of tuples (common from direct DB execution)
        if isinstance(raw_result, list) and len(raw_result) > 0:
            # Check if it's a list of tuples
            if all(isinstance(item, tuple) for item in raw_result):
                if len(raw_result[0]) == 2:
                    # Handle age-specific data
                    if all(isinstance(item[0], int) and 10 <= item[0] <= 100 for item in raw_result):
                        return pd.DataFrame(raw_result, columns=["Age", "AveragePurchaseAmount"])
                    # Common case for customer or product data with ID and value
                    col1_name = "Customer ID" if any(str(x[0]).startswith("CUST") for x in raw_result) else "ID"
                    col2_name = "Amount" if any(isinstance(x[1], (Decimal, float, int)) for x in raw_result) else "Value"
                    return pd.DataFrame(raw_result, columns=[col1_name, col2_name])
                else:
                    # Generate generic column names for other cases
                    cols = [f"Column {i+1}" for i in range(len(raw_result[0]))]
                    return pd.DataFrame(raw_result, columns=cols)
        
        # Handle string representation of results (alternative approach)
        if isinstance(raw_result, str):
            # Try to extract tuple data from string representation
            import re
            
            # Check if it's a list of tuples pattern
            pattern = r"\((\d+),\s*Decimal\('([^']+)'\)\)"
            matches = re.findall(pattern, raw_result)
            
            if matches:
                # Extract age and amounts
                data = [(int(age), float(amount)) for age, amount in matches]
                return pd.DataFrame(data, columns=["Age", "AveragePurchaseAmount"])
            
            # Check for customer ID pattern
            pattern = r"\('([^']+)',\s*Decimal\('([^']+)'\)\)"
            matches = re.findall(pattern, raw_result)
            
            if matches:
                # Extract customer IDs and amounts
                data = [(cust_id, float(amount)) for cust_id, amount in matches]
                col1_name = "Customer ID" if any(id.startswith("CUST") for id, _ in data) else "ID"
                return pd.DataFrame(data, columns=[col1_name, "Amount"])
            
            # If it doesn't match our pattern, return as a single result
            return pd.DataFrame([{"Result": raw_result}])
        
        # For empty or unhandled results
        return pd.DataFrame()
        
        def format_value(self, value):
            """Format values nicely for display"""
            if isinstance(value, (Decimal, float)) and value >= 10:
                return f"${value:,.2f}"
            elif isinstance(value, (int, Decimal, float)):
                return f"{value:,}"
            return str(value)
    
    def format_text_output(self, df, question):
        """Format DataFrame results into clean text format"""
        if df is None or df.empty:
            return "No results found for your query."
        
        # Generate header based on question type
        if any(word in question.lower() for word in ["top", "highest", "best", "most"]):
            header = f"**Top {len(df)} Results:**"
        elif any(word in question.lower() for word in ["how many", "count", "number of"]):
            header = f"**Count Results ({len(df)} items):**"
        elif any(word in question.lower() for word in ["average", "mean", "avg"]):
            header = f"**Average Values ({len(df)} items):**"
        elif any(word in question.lower() for word in ["total", "sum", "overall"]):
            header = f"**Total Values ({len(df)} items):**"
        else:
            header = f"**Query Results ({len(df)} items):**"
        
        # Create markdown table
        table_header = "| "
        table_separator = "| "
        
        # Build table header and separator
        for col in df.columns:
            formatted_col = col.replace("_", " ").title()
            table_header += f"{formatted_col} | "
            table_separator += "--- | "
        
        # Build table rows
        table_rows = ""
        for _, row in df.iterrows():
            table_row = "| "
            for value in row:
                formatted_value = self.format_value(value)
                table_row += f"{formatted_value} | "
            table_rows += table_row + "\n"
        
        # Combine into markdown table
        markdown_table = f"{header}\n\n{table_header}\n{table_separator}\n{table_rows}"
        
        return markdown_table
    
    def execute_query(self, question):
        """Process natural language query and return formatted results"""
        try:
            # Create SQL query chain
            chain = create_sql_query_chain(self.llm, self.db)

            # Generate SQL query
            generated_response = chain.invoke({"question": question})
            generated_query = self.extract_sql_query(generated_response)

            # Execute query
            raw_result = self.db.run(generated_query)

            # Process result
            result_df = self.parse_result(raw_result)

            # Determine result type for title formatting
            result_type = "Results"
            if any(word in question.lower() for word in ["top", "highest", "best", "most"]):
                count = len(result_df) if isinstance(result_df, pd.DataFrame) else 1
                result_type = f"Top {count} Results"

            return {
                "query": generated_query,
                "raw_result": raw_result,
                "result_df": result_df,
                "result_type": result_type,
                "error": None
            }
        except Exception as e:
            import traceback
            error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            return {
                "query": None,
                "raw_result": None,
                "result_df": None,
                "result_type": None,
                "error": error_msg
            }

# ========== UI COMPONENTS ==========
class RetailAppUI:
    def __init__(self):
        self.setup_page_config()
        self.apply_custom_styles()
        
    def setup_page_config(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="Retail Sales Query App",
            page_icon="🤖",
            layout="wide",
        )
    
    def apply_custom_styles(self):
        """Apply custom CSS styles"""
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem !important;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 0;
        }
        .sub-header {
            font-size: 1.2rem !important;
            color: #7f8c8d;
            margin-top: 0;
        }
        # .query-card {
        #     background-color: #000000;
        #     padding: 20px;
        #     border-radius: 10px;
        #     border-left: 5px solid #4285f4;
        #     margin-bottom: 20px;
        }
        .results-header {
            font-size: 1.5rem !important;
            font-weight: bold;
            color: #2c3e50;
            margin-top: 20px;
        }
        # .result-card {
        #     background-color: #f8f9fa;
        #     border-left: 5px solid #34a853;
        #     padding: 15px 20px;
        #     border-radius: a5px;
        #     margin: 15px 0;
        #     line-height: 1.5;
        # }
        .example-btn {
            background-color: #f1f3f4;
            border: 1px solid #dadce0;
            color: #202124;
            padding: 8px 16px;
            border-radius: 4px;
            margin: 4px;
            cursor: pointer;
            text-align: left;
            transition: all 0.3s;
        }
        .example-btn:hover {
            background-color: #e8f0fe;
            border-color: #4285f4;
        }
        .sql-header {
            font-size: 1rem !important;
            font-weight: bold;
            color: #2c3e50;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def render_header(self):
        """Render application header"""
        col1, col2 = st.columns([1, 6])
        with col1:
            st.markdown("# 🤖")
        with col2:
            st.markdown('<p class="main-header">SQL-BOT</p>', unsafe_allow_html=True)
            st.markdown('<p class="sub-header">Ask questions about your data in plain English</p>', unsafe_allow_html=True)
    
    def render_sidebar(self, db_name, available_tables):
        """Render application sidebar"""
        with st.sidebar:
            st.header("About This App")
            st.markdown("""
            This app lets you query your retail sales database using natural language. 
            Just type your question, and our AI will convert it to SQL and fetch the results.
            """)
            
            st.subheader("Database Connection")
            st.success("✅ Connected to database")
            st.info(f"Database: {db_name}")
            
            with st.expander("Available Tables"):
                st.write(available_tables)
    
    def render_query_section(self, current_question=""):
        """Render query input area and examples"""
        st.markdown('<div class="query-card">', unsafe_allow_html=True)
        
        # Query input
        question = st.text_area(
            "💬 Type your question here:",
            value=current_question,
            height=100,
            placeholder="For example: Which products were most popular last month?"
        )
        
        # Action buttons
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            execute_button = st.button("🔍 Get Answer", type="primary", use_container_width=True)
        with col2:
            clear_button = st.button("🗑 Clear", use_container_width=True)
            
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Example questions
        st.markdown("#### Try one of these examples:")
        
        example_questions = [
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
        
        # Display example questions in a grid
        cols = st.columns(3)
        for i, example in enumerate(example_questions):
            col_idx = i % 3
            with cols[col_idx]:
                if st.button(f"📝 {example}", key=f"example_{i}", 
                           help=example, use_container_width=True):
                    question = example
                    execute_button = True
        
        return question, execute_button, clear_button
    
    def render_results(self, results):
        """Render query results"""
        if results is None:
            return
            
        if results.get("error"):
            st.error(results["error"])
            return
            
        # SQL Query Display
        if results.get("query"):
            with st.expander("💻 View SQL Query", expanded=False):
                st.markdown('<p class="sql-header">Generated SQL:</p>', unsafe_allow_html=True)
                st.code(results["query"], language="sql")
        
        # Results Display
        if results.get("result_df") is not None and not results["result_df"].empty:
            st.markdown('<p class="results-header">📊 Results</p>', unsafe_allow_html=True)
            
            # Always display as a styled Streamlit table first
            df = results["result_df"]
            
            # Format the amounts if they exist
            if "Amount" in df.columns:
                df["Amount"] = df["Amount"].apply(lambda x: f"${x:,.2f}" if isinstance(x, (float, int, Decimal)) else x)
            
            # Custom styling for table
            st.dataframe(
                df,
                use_container_width=True,
                height=min(35 * len(df) + 38, 400)
            )
            
            # Download option
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="query_results.csv",
                mime="text/csv",
            )
        elif results.get("result_df") is not None:
            st.info("No data found for your query. Try asking a different question.")

# ========== MAIN APPLICATION ==========
def main():
    # Database connection parameters
    db_user = "root"
    db_password = "root"
    db_host = "localhost"
    db_name = "retail_sales_db"
    
    # Initialize UI
    ui = RetailAppUI()
    ui.render_header()
    
    # Initialize Model
    try:
        model = RetailQueryModel(db_user, db_password, db_host, db_name)
        ui.render_sidebar(db_name, model.db.get_usable_table_names())
    except Exception as e:
        st.error(f"Failed to initialize application: {str(e)}")
        st.stop()
    
    # Handle session state
    if "question" not in st.session_state:
        st.session_state.question = ""
    if "results" not in st.session_state:
        st.session_state.results = None
    
    # Render query section
    question, execute_button, clear_button = ui.render_query_section(st.session_state.question)
    
    # Handle clear button
    if clear_button:
        st.session_state.question = ""
        st.session_state.results = None
        st.rerun()
    
    # Handle execute button
    if execute_button and question.strip():
        st.session_state.question = question
        
        with st.spinner("🔄 Finding the answer to your question..."):
            results = model.execute_query(question)
            st.session_state.results = results
    
    # Render results
    ui.render_results(st.session_state.results)

if __name__ == "__main__":
    main()