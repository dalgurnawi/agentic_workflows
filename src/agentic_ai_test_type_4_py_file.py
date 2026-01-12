# Import packages
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
from typing import Union, List, Dict
import datetime
from datetime import date 
import csv
import ast
from typing import Optional
import signal
from pathlib import Path
import time
from sqlalchemy import create_engine

# model
llm_model = "deepseek-r1:14b"

x = datetime.datetime.now()

# Langchain API KEY
LANGSMITH_API_KEY=os.getenv('LANGSMITH_API_KEY')
LANGSMITH_ENDPOINT=os.getenv('LANGSMITH_ENDPOINT')
# DATABASE Connection settings
DB_NAME=os.getenv('DB_NAME')
USERNAME=os.getenv('USERNAME')
PASSWORD=urllib.parse.quote(os.getenv('PASSWORD'))
HOSTNAME=os.getenv('HOSTNAME')
PORT=os.getenv('PORT')

# Creating conneciton to database

engine = create_engine(f'postgresql+psycopg2://{USERNAME}:{PASSWORD}@{HOSTNAME}/{DB_NAME}')
#conn = psycopg2.connect(f"dbname={DB_NAME} user={USERNAME} password={PASSWORD}")
accounts_receivables = pd.read_sql("SELECT * FROM accounts_receivable", engine)
payments= pd.read_sql("SELECT * FROM payments", engine)


class TimeoutException(Exception):   # Custom exception class
    pass

def timeout_handler(signum, frame):   # Custom signal handler
    raise TimeoutException

# Change the behavior of SIGALRM
signal.signal(signal.SIGALRM, timeout_handler)

# Add memory to your agent to maintain state across interactions. This allows the agent to remember previous conversations and context.
checkpointer = InMemorySaver()

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

model = ChatOllama(
    model=llm_model,
    temperature=0,
    disable_streaming=True,
    #format="json", # Enforce JSON Schema
    #num_gpu=1
) 

def row_gen(dataframe):
    for _, row in dataframe.iterrows():
        yield str(row.to_dict())

generator_ar = row_gen(accounts_receivables)

generator_p= row_gen(payments)

def safe_eval_with_dates(string_data):
    """
    Safely evaluate a string containing datetime objects and escaped quotes.
    Converts datetime.date() calls to ISO format strings and fixes escaping.
    """
    import re
    from datetime import date
    
    # First, fix escaped quotes - replace \' with just '
    # ast.literal_eval expects proper Python string literals
    cleaned = string_data.replace("\\'", "'")
    
    # Replace datetime.date(year, month, day) with ISO format strings
    pattern = r'datetime\.date\((\d+),\s*(\d+),\s*(\d+)\)'
    
    def replace_date(match):
        year, month, day = match.groups()
        try:
            d = date(int(year), int(month), int(day))
            return f'"{d.isoformat()}"'
        except ValueError:
            return match.group(0)  # Return original if invalid date
    
    cleaned = re.sub(pattern, replace_date, cleaned)
    
    # Now use ast.literal_eval on the cleaned string
    return ast.literal_eval(cleaned)

def error_log(model, error_message, ar_row_affected, p_row_affected):
    x = datetime.datetime.now()
    log_file = Path(f"updated_ar_error_log_{model}_run_{x.day}-{x.month}-{x.year}.csv")
    log_file.touch(exist_ok=True)
    f = open(log_file, 'w')
    f.write(f"Time: {x.hour}:{x.minute}\nError message: {error_message}\nAR Row affected: {ar_row_affected}\nAR Datatype {type(ar_row_affected)}\nPayment Row affected: {p_row_affected})\nPayment Datatype {type(p_row_affected)}")
    f.close()

def turn_based_checker(accounts_receivable, payments, generator_ar, generator_p):
    """ Match apyments to accounts receivables and update AR records in an iterative manner
        Args:
            accounts_receivable: DataFrame of AR invoices
            payments: DataFrame of payments
            generator_ar: Method to return the next row of AR as a dictionary
            generator_p: Method to return the next row of payments as a dictionary
            Checkpointer: Checkpointer for the agent.
        
        Returns:
            DataFrame: Updated accounts receivable records
    """
    updated_ar = pd.DataFrame(columns=accounts_receivable.columns)
    counter = 1600
    for i in range(len(accounts_receivable)+1):
        print(f"Processing Invoice Nr: {i+1}/ {len(accounts_receivable)}")
        # Get next AR row
        try:
            row_ar = next(generator_ar)
        except StopIteration: # Otherwise it will through an error if the ARs have been cycled through
            break
        
        if row_ar is None:
            break

        is_match = False
        
        generator_p= row_gen(payments) # Recreate the payments generator, so that we do not run into a Stopiteration Error
        # Stop iteration error is raised when a generator is exhausted and will have to be created again
        
        for j in tqdm(range(len(payments)), desc=f"Matching payments for invoice {i+1}"):
            try:
                row_p = next(generator_p)
            except StopIteration:
                break

            if row_p is None:
                continue
            # Run agent
            #`thread_id` is a unique identifier for a given conversation
            config = {"configurable": {"thread_id": f"{counter}"}}
            # Create the agent with default parameters
            agent = create_agent(model=model, system_prompt=prompt)
         
            # Start signal timer to kill the process if it hangs longer than 5 mins
            signal.alarm(900)

            try:
            
                # Ask the agent a question
                response = agent.invoke({"messages": [{"role": "user", 
                                "content": f"""
                You have been given an accounts receivable invoice {row_ar} and a received payment {row_p}. Assess whether the payment relates to the invoice. 
                If there is no match, return an empty dictionary. If it does, UPDATE the invoice data with relevant payment information and return 
                the updated record as a dictionary with all invoice fields. Do not create any new columns for accounts receivables. Return the updated accounts receivables data as
                a dictionary with no other additional text.
                """}]},    config=config,
                    context=Context())
            
            except TimeoutException as e:
                error_log(model=llm_model, error_message=e, ar_row_affected=row_ar, p_row_affected=row_p)
                continue # Continue the loop if function takes longer than 5 mins

            else:
                #Reset the alarm
                signal.alarm(0)

            counter +=1
            ai_messages = [
                m for m in response["messages"]
                if isinstance(m, AIMessage)
            ]
            #print(ai_messages)
            output_list=[message.content for message in ai_messages]

            #print(output_list)
            output_list = [x for x in output_list if not ('think' in x or x == '')]
            
            try:
                output_list = [safe_eval_with_dates(x) for x in output_list]
                
            except Exception as e:
                e = f"Error at transformation to dictionary: {e}"
                error_log(model=llm_model, error_message=e, ar_row_affected=row_ar, p_row_affected=row_p)
                continue

            non_empty = 0
            empty = 0
            for item in output_list:
                if item:
                    non_empty +=1
                else:
                    empty+=1

            if non_empty >= 1: # We know that there has been at least one match
                is_match = True
                for output in output_list:
                    try:
                        if output is None: # Make sure only Non-Null rows are added
                            pass
                        else:
                            updated_row = pd.DataFrame(output, index=[counter])
                            updated_ar = pd.concat([updated_ar, updated_row], ignore_index=True)
                            #print(f"Updated AR after match found: {updated_ar}")
                    except Exception as e:
                        error_log(model=llm_model, error_message=e, ar_row_affected=row_ar, p_row_affected=row_p)

            else:
                pass
           
        if not is_match:
            try:
                old_ar_row = safe_eval_with_dates(row_ar)
                updated_ar = pd.concat([updated_ar, old_ar_row.to_frame().T], ignore_index=True)
                #print(f"Updated AR after no match: {updated_ar}")
            except Exception as e:
                error_log(model=llm_model, error_message=e, ar_row_affected=row_ar, p_row_affected=row_p)
                continue

        else:
            pass
    # Save updated DF to csv
    updated_ar.dropna(axis=0, how='all', inplace=True)
    updated_ar.drop_duplicates(keep='first', inplace=True, ignore_index=True)
    updated_ar.to_csv(f'updated_ar_{llm_model}_run_{x.day}-{x.month}-{x.year}.csv', index=False)   

if __name__ == "__main__":
    start = time.perf_counter()
    turn_based_checker(accounts_receivable=accounts_receivables, payments=payments, generator_ar=generator_ar, generator_p=generator_p)
    end = time.perf_counter()
    print(f"Elapsed time: {end - start:.3f} seconds")
