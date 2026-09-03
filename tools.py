import re
import sqlite3
import pandas as pd
from langchain_core.tools import tool

DB_PATH = "data/sales.db"


@tool
def get_database_schema() -> str:
    """Get the schema (tables and columns) of the sales database."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cur.fetchall()]

    schema_parts = []
    for table in tables:
        cur.execute(f"PRAGMA table_info({table});")
        columns = cur.fetchall()
        col_desc = ", ".join([f"{col[1]} ({col[2]})" for col in columns])
        schema_parts.append(f"Table '{table}': {col_desc}")

    conn.close()
    return "\n".join(schema_parts)


@tool
def execute_sql(query: str) -> str:
    """
    Execute a READ-ONLY SQL query against the sales database and return the results.
    Only SELECT and WITH statements are allowed. Any other statement will be rejected.
    """
    forbidden = ["insert", "update", "delete", "drop", "alter", "truncate", "create"]
    lowered = query.strip().lower()

    if not (lowered.startswith("select") or lowered.startswith("with")):
        return "Error: Only SELECT or WITH statements are allowed."

    pattern = r'\b(' + '|'.join(forbidden) + r')\b'
    if re.search(pattern, lowered):
        return "Error: Query contains a forbidden keyword. Only read-only SELECT queries are allowed."

    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df.to_string(index=False)
    except Exception as e:
        return f"Error executing query: {e}"


if __name__ == "__main__":
    print(get_database_schema.invoke({}))
    print()
    print(execute_sql.invoke({"query": "SELECT * FROM products LIMIT 3;"}))