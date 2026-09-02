import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage, SystemMessage, trim_messages
from tools import get_database_schema, execute_sql


class TrimHistory(AgentMiddleware):
    """Trim the conversation to a token budget before each model call.

    The free Groq tier caps tokens-per-minute quite low, and every tool round
    resends the whole history, so we keep only the most recent messages to stay
    under the limit.
    """

    def before_model(self, state, runtime):
        messages = state["messages"]
        if len(messages) <= 2:
            return None
        trimmed = trim_messages(
            messages,
            max_tokens=6000,
            token_counter="approximate",
            strategy="last",
            include_system=True,
            start_on=HumanMessage,
        )
        return {"messages": trimmed}

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
    middleware=[TrimHistory()],
)


if __name__ == "__main__":
    question = "What was the total revenue in July 2026?"
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})

    # Print the final answer
    final_message = result["messages"][-1]
    print("\nFINAL ANSWER:\n")
    print(final_message.content)