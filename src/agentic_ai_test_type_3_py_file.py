import json
from enum import Enum
from pathlib import Path
import psycopg2
import ollama
import pandas as pd
from IPython.display import Image, Markdown, display
from tqdm import tqdm
import os
from dotenv import load_dotenv, find_dotenv
import urllib.parse
from langchain.messages import AIMessage
from langchain_ollama import ChatOllama
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_agent
from agentic_ai_tools import AccessAccountsReceivable, AccessPayments
import datetime as dt
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.structured_output import ToolStrategy
from langchain_experimental.agents import create_pandas_dataframe_agent
from typing import Union, List, Dict, Optional
import datetime
from sqlalchemy import create_engine
import re
import time

# ENV Variables
# Langchain API KEY
LANGSMITH_API_KEY=os.getenv('LANGSMITH_API_KEY')
LANGSMITH_ENDPOINT=os.getenv('LANGSMITH_ENDPOINT')
# DATABASE Connection settings
DB_NAME=os.getenv('DB_NAME')
USERNAME=os.getenv('USERNAME')
PASSWORD=urllib.parse.quote(os.getenv('PASSWORD'))
HOSTNAME=os.getenv('HOSTNAME')
PORT=os.getenv('PORT')
engine = create_engine(f'postgresql+psycopg2://{USERNAME}:{PASSWORD}@{HOSTNAME}/{DB_NAME}')

# Today's date
x = datetime.datetime.now()

# LLM Model:
llm_model = "gpt-oss:20b"

# Create data models

class Context():
    """Custom runtime context schema."""
    user_id: str
class Invoice(BaseModel):
    invoice_number: Optional[int] = None
    date: Optional[str] = None
    customer_name: Optional[str] = None
    customer_number: Optional[int] = None
    amount: Optional[float] = None
    due_date: Optional[str] = None
    payment: Optional[float]= None
    payment_date: Optional[str] = None
    payment_id: Optional[int] = None

prompt ="""
You are a senior accountant responsible for accounts receivable reconciliation.

You have access to the following tools:
- AccessAccountsReceivable: returns the current accounts receivable as structured data.
- AccessPayments: returns all received payments as structured data.

Rules:
- When reconciliation or invoice updates are requested, you MUST call the tools.
- Always call AccessAccountsReceivable BEFORE AccessPayments.
- Never ask the user follow-up questions.
- Do not explain your reasoning.
- Do not return stringified data.
- Do not return data with different keys.

Output requirements:
- Return ONLY a Python dictionary.
- The dictionary must represent the UPDATED accounts receivable.
- Do not wrap the output in text, markdown, or code fences.
"""
# Call dataframes and return them as dictionaries
accounts_receivables = pd.read_sql("SELECT * FROM accounts_receivable", engine)
dict_ar = accounts_receivables.to_dict()
payments = pd.read_sql("SELECT * FROM payments", engine)
dict_p = payments.to_dict()

# Add memory to your agent to maintain state across interactions. This allows the agent to remember previous conversations and context.
checkpointer = InMemorySaver()

tools=[AccessAccountsReceivable, AccessPayments]
model = ChatOllama(
    model=llm_model,
    temperature=0,
) 

#`thread_id` is a unique identifier for a given conversation
config = {"configurable": {"thread_id": f"{x.hour}{x.minute}"}}
# Create the agent with default parameters
agent = create_agent(model=model, system_prompt=prompt, tools=tools, checkpointer=checkpointer)

if __name__ == "__main__":
    start = time.perf_counter()

    # Run agent
    response = agent.invoke({"messages": [{"role": "user", 
                    "content": f"""
                    Reconcile received payments against the current accounts receivableand update invoice statuses accordingly. 
                    Return the updated accounts receivable, which should be the a dictionary in the same format of the accounts receivables with the data updated for matched invoices
                    and the original data for invoices that have not been matched.
    """}]},    config=config,
        context=Context(), response_format=Invoice)

    end = time.perf_counter()
    print(f"Elapsed time: {end - start:.3f} seconds")

    ai_messages = [
    m for m in response["messages"]
    if isinstance(m, AIMessage)
    ]
    #print(ai_messages)
    output_list=[message.content for message in ai_messages]
    output_list