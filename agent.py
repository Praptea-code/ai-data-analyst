import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from tools import get_database_schema, execute_sql

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)

SYSTEM_PROMPT = """You are a careful data analyst assistant with access to a sales database.

You have two tools:
- get_database_schema: use this first to understand what tables and columns exist.
- execute_sql: use this to run READ-ONLY SQL queries (SELECT/WITH only) against the database.

Always check the schema before writing SQL if you haven't already seen it.
Answer the user's question clearly using the actual data returned by execute_sql.
Never make up numbers — only use what the tool returns.
"""

agent = create_agent(
    model=llm,
    tools=[get_database_schema, execute_sql],
    system_prompt=SYSTEM_PROMPT,
)


if __name__ == "__main__":
    question = "What was the total revenue in July 2026?"
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})

    # Print the final answer
    final_message = result["messages"][-1]
    print("\nFINAL ANSWER:\n")
    print(final_message.content)