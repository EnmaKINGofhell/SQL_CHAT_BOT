import os
import pandas as pd
import re
import ast
from decimal import Decimal
from sqlalchemy import create_engine, text
from langchain.chains import create_sql_query_chain
from langchain_community.utilities import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

class IntegratedRetailQueryModel:
    def __init__(self, db_user, db_password, db_host, db_name, use_llm=True):
        self.db_user = db_user
        self.db_password = db_password
        self.db_host = db_host
        self.db_name = db_name
        self.use_llm = use_llm
        self.db = self._connect_to_database()
        self.llm = self._initialize_llm() if use_llm else None

    def _connect_to_database(self):
        try:
            engine = create_engine(f"mysql+pymysql://{self.db_user}:{self.db_password}@{self.db_host}/{self.db_name}")
            return SQLDatabase(engine, sample_rows_in_table_info=5)
        except Exception as e:
            raise ConnectionError(f"Database connection failed: {e}")

    def _initialize_llm(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        return ChatGoogleGenerativeAI(
            model="models/gemini-1.5-pro",
            temperature=0.1,
            google_api_key=api_key
        )

    def execute_query(self, question):
        try:
            # Generate SQL query
            chain = create_sql_query_chain(self.llm, self.db)
            generated_response = chain.invoke({"question": question})
            generated_query = self._extract_sql_query(generated_response)
            
            # Execute query
            raw_result = self.db.run(generated_query)
            
            # Process and format results
            result_df = self._parse_result(raw_result, question)
            formatted_text = self._format_text_output(result_df, question)
            
            return {
                "query": generated_query,
                "raw_result": raw_result,
                "result_df": result_df,
                "formatted_text": formatted_text,
                "error": None
            }
        except Exception as e:
            return {
                "query": None,
                "raw_result": None,
                "result_df": None,
                "formatted_text": None,
                "error": str(e)
            }

    def _extract_sql_query(self, response):
        if "```sql" in response:
            return response.split("```sql")[1].split("```")[0].strip()
        return response

    def _parse_result(self, raw_result, question):
        # Handle case where raw_result is already a DataFrame
        if isinstance(raw_result, pd.DataFrame):
            return raw_result
            
        # Handle string representation of tuples
        if isinstance(raw_result, str) and raw_result.startswith("[("):
            try:
                raw_result = ast.literal_eval(raw_result)
            except:
                pass
                
        # Handle list of tuples
        if isinstance(raw_result, list) and raw_result and isinstance(raw_result[0], tuple):
            # Handle case where we have a list containing one tuple of results
            if len(raw_result) == 1 and isinstance(raw_result[0][0], tuple):
                raw_result = raw_result[0]
                
            # Determine column names based on question
            columns = self._determine_column_names(raw_result, question)
            return pd.DataFrame(raw_result, columns=columns)
            
        # Default case - create DataFrame with single column
        return pd.DataFrame([{"Result": raw_result}])

    def _determine_column_names(self, result, question):
        question_lower = question.lower()
        num_columns = len(result[0]) if result else 0
        
        # Customer spending pattern
        if "customer" in question_lower and "spend" in question_lower and num_columns == 2:
            return ["Customer ID", "Total Spending"]
        # Product sales pattern
        elif "product" in question_lower and "sales" in question_lower and num_columns == 2:
            return ["Product", "Sales Amount"]
        # Top N pattern
        elif "top" in question_lower and num_columns == 2:
            return ["Item", "Value"]
        # Default column names
        return [f"Column {i+1}" for i in range(num_columns)]

    def _format_value(self, value):
        if value is None:
            return "N/A"
        if isinstance(value, (Decimal, float)):
            if abs(value) >= 10:
                return f"${value:,.2f}"
            return f"{value:,.2f}"
        if isinstance(value, int):
            return f"{value:,}"
        return str(value)

    def _format_text_output(self, df, question):
        if df.empty:
            return "No results found for your query."
            
        question_lower = question.lower()
        num_rows = len(df)
        num_cols = len(df.columns)
        
        # Generate appropriate header
        if any(term in question_lower for term in ["top", "highest", "most"]):
            if num_cols == 2:
                if num_rows == 1:
                    return f"Top result: {df.iloc[0,0]} with {self._format_value(df.iloc[0,1])}"
                return f"Top {num_rows} results:\n" + "\n".join(
                    f"- {row[0]}: {self._format_value(row[1])}" for row in df.itertuples(index=False)
                )
        
        # Default formatting for other cases
        if num_cols == 1:
            return f"Result: {self._format_value(df.iloc[0,0])}" if num_rows == 1 else \
                   f"Found {num_rows} items:\n" + "\n".join(f"- {self._format_value(val)}" for val in df.iloc[:,0])
        
        if num_cols == 2:
            return "\n".join(f"- {row[0]}: {self._format_value(row[1])}" for row in df.itertuples(index=False))
        
        # For more than 2 columns
        return "\n".join(
            f"- {', '.join(f'{col}: {self._format_value(val)}' for col, val in zip(df.columns, row))"
            for row in df.itertuples(index=False)
        )