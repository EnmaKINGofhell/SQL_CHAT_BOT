#!/usr/bin/env python
# coding: utf-8

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_sql_query_chain
from langchain_community.utilities import SQLDatabase
from pymysql.err import ProgrammingError

# ---------- Load Environment Variables ----------
load_dotenv(dotenv_path="env.txt")  # Load API key from env.txt
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY not found in env.txt")

# ---------- Set up LLM ----------
llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-pro", 
    temperature=0.3,
    google_api_key=GOOGLE_API_KEY
)

# ---------- Connect to MySQL ----------
db_user = "root"
db_password = "root"
db_host = "localhost"
db_name = "retail_sales_db"

db = SQLDatabase.from_uri(
    f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}",
    sample_rows_in_table_info=3
)

print("✅ Connected to database")
print("\n🔍 Tables:\n", db.get_usable_table_names())

# ---------- Build SQL Query Chain ----------
chain = create_sql_query_chain(llm, db)

# ---------- Query Execution Function ----------
def execute_query(question):
    try:
        print(f"\n🧠 Question: {question}")

        # Generate SQL query from question
        response = chain.invoke({"question": question})
        print("📜 Raw Response:\n", response)

        # 🛠 Extract SQL query
        cleaned_query = None

        # Case 1: Inside ```sql ... ```
        if "```sql" in response:
            cleaned_query = response.split("```sql")[1].split("```")[0].strip()
        
        # Case 2: Contains "SQLQuery:" prefix
        elif "SQLQuery:" in response:
            cleaned_query = response.split("SQLQuery:")[1].strip().split("\n")[0]
        
        # Case 3: First SELECT/UPDATE/etc. line fallback
        else:
            lines = response.splitlines()
            for line in lines:
                if line.strip().upper().startswith(("SELECT", "UPDATE", "DELETE", "INSERT")):
                    cleaned_query = line.strip()
                    break

        if not cleaned_query:
            raise ValueError("⚠️ Could not extract SQL query from LLM response.")

        print("✅ Cleaned SQL:\n", cleaned_query)

        # Execute the cleaned SQL query
        result = db.run(cleaned_query)
        print("📊 Result:\n", result)

    except ProgrammingError as e:
        print(f"⚠️ SQL Error: {e}")
    except Exception as e:
        print(f"⚠️ Unexpected Error: {e}")


# ---------- Sample Questions ----------
questions = [
    "How many unique customers are there for each product category?",
    "Calculate total sales amount per product category.",
    "Calculate the average age of customers grouped by gender.",
    "Identify the top spending customers based on their total amount spent.",
    "Count the number of transactions made each month.",
    "Calculate the total sales amount and average price per unit for each product category."
]

# ---------- Run all queries ----------
for q in questions:
    execute_query(q)
    print("--------------------------------------------------")
