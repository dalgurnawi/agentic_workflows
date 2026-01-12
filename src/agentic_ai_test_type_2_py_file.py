# Import packages
from enum import Enum
from pathlib import Path
import psycopg2
import pandas as pd
from IPython.display import Image, Markdown, display
from tqdm import tqdm
import os
from langchain.messages import AIMessage
from langchain_ollama import ChatOllama
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_agent
from agentic_ai_tools import PaymentReferenceSearch, AccessAccountsReceivable, AccessPayments
import datetime as dt
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.structured_output import ToolStrategy
import urllib.parse
from sqlalchemy import create_engine
from typing import Optional
from datetime import date
import datetime
import time


# ENV variables

# Langchain API KEY
LANGSMITH_API_KEY=os.getenv('LANGSMITH_API_KEY')
LANGSMITH_ENDPOINT=os.getenv('LANGSMITH_ENDPOINT')
# DATABASE Connection settings
DB_NAME=os.getenv('DB_NAME')
USERNAME=os.getenv('USERNAME')
PASSWORD=urllib.parse.quote(os.getenv('PASSWORD'))
HOSTNAME=os.getenv('HOSTNAME')
PORT=os.getenv('PORT')

# Model selection
llm_model = "gpt-oss:20b"

# Today's date
x = datetime.datetime.now()

# Pydantic data models
class Context():
    """Custom runtime context schema."""
    user_id: str

class ARResponse(BaseModel):
    """Accounts Receivable dataframe content"""
    invoice_number: int 
    date: dt.date
    customer_name: str
    customer_number: int
    amount: float 
    due_date: dt.date 
    payment: float | None = None
    payment_date: dt.date | None = None
    payment_id: int | None = None

class PayResponse(BaseModel):
    "Payments dataframe content"
    transaction_id: int
    payment_date: dt.date 
    payment_amount: float 
    payment_reference: str | None = None
    matched: bool

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

prompt =f"""You are a senior accountant responsible for accounts receivable reconciliation.

Rules:
- When reconciliation or invoice updates are requested, you determine whether the payment is related to the invoice.
- Never ask the user follow-up questions.
- Do not explain your reasoning.
- Do not return stringified data.
- Do not return different structured data.
- DO NOT nest the payments data into the updated accounts receivable information.

Output requirements:
- Return ONLY a Python dictionary.
- The dictionary must represent the UPDATED accounts receivable.
- Do not wrap the output in text, markdown, or code fences.
            """

# Add memory to your agent to maintain state across interactions. This allows the agent to remember previous conversations and context.
checkpointer = InMemorySaver() # Keeps Agent stateful

tools=[PaymentReferenceSearch, AccessAccountsReceivable, AccessPayments]
model = ChatOllama(
    model=llm_model,
    temperature=0,
) 
# model_with_tools = model.bind_tools(tools=tools) # only for OpenAI
#agent = create_agent(model=model, system_prompt=prompt, tools=tools, response_format=ToolStrategy(Union[ARResponse, PayResponse]), checkpointer=checkpointer)
agent = create_agent(model=model, system_prompt=prompt, tools=tools, checkpointer=checkpointer)


if __name__ == "__main__":
    start = time.perf_counter()
    # Run agent
    #`thread_id` is a unique identifier for a given conversation
    config = {"configurable": {"thread_id": f"{x.hour}{x.minute}"}}

    response = agent.invoke(
        {"messages": [{"role": "user", 
                            "content": """ Without responding with questions and using the tools provided perform the following steps in order:
                            1. Access the accounts receivables as a pandas dataframe using the AccessPayments tool,
                            2. Access the payments received as a pandas dataframe AccessAccountsReceivable tool,
                            3. Match the payments received to the open accounts receivables using the PaymentReferenceSearch tool provided,
                            4. If there is no match, return an empty dictionary. If it does, UPDATE the invoice data with relevant payment information and return 
                            the updated record as a dictionary with all invoice fields using the following schema:
                            invoice_number: Optional[int] = None
                            date: Optional[str] = None
                            customer_name: Optional[str] = None
                            customer_number: Optional[int] = None
                            amount: Optional[float] = None
                            due_date: Optional[str] = None
                            payment: Optional[float]= None
                            payment_date: Optional[str] = None
                            payment_id: Optional[int] = None 
                
                Do not create any new columns for accounts receivables. Return the updated accounts receivables data as
                        a dictionary with no other additional text."""}]},
        config=config,
        context=Context(),
        response_format=Invoice
    )
    end = time.perf_counter()
    print(f"Elapsed time: {end - start:.3f} seconds")