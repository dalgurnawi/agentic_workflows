# Import packages
import ast
import json
from enum import Enum
from pathlib import Path
import psycopg2
import ollama
import pandas as pd
from IPython.display import Image, Markdown, display
from tqdm import tqdm
import os
import urllib.parse
from sqlalchemy import create_engine
import time

# ENV Variables
MODEL = "gpt-oss:20b"
TEMPERATURE = 0
DB_NAME=os.getenv('DB_NAME')
USERNAME=os.getenv('USERNAME')
PASSWORD=urllib.parse.quote(os.getenv('PASSWORD')) # Need to parse, since password has special characters
HOSTNAME=os.getenv('HOSTNAME')
PORT=os.getenv('PORT')

# Creating conneciton to database
engine = create_engine(f'postgresql+psycopg2://{USERNAME}:{PASSWORD}@{HOSTNAME}/{DB_NAME}')

# Connect to accounts receivables table and return dataframe
accounts_receivable = pd.read_sql("SELECT * FROM accounts_receivable", engine)

# Connect to payments and return datframe
payments= pd.read_sql("SELECT * FROM payments", engine)

# Create Data Classes to return responses as

class ResponseFormat(Enum):
    JSON = "json_object"
    TEXT = "text"

# Function to call model
def call_model(
    prompt: str, response_format: ResponseFormat = ResponseFormat.TEXT
) -> str:
    response = ollama.generate(
        model=MODEL,
        prompt=prompt,
        keep_alive="1h",
        format="" if response_format == ResponseFormat.TEXT else "json",
        options={"temperature": TEMPERATURE},
    )
    return response["response"]

task = f""" You have been given an accounts receivable invoice {accounts_receivable} and a received payment {payments}. Assess whether the payment relates to the invoice. 
            If there is no match, return an empty dictionary. If it does, UPDATE the invoice data with relevant payment information and return 
            the updated record as a dictionary with all invoice fields. Do not create any new columns for accounts receivables. Return the updated accounts receivables data as
            a dictionary with no other additional text."""


if __name__ == "__main__":
    start = time.perf_counter()
    # Call model
    response = call_model(task)
    end = time.perf_counter()
    print(f"Elapsed time: {end - start:.3f} seconds")

    data = json.loads(response)
    print(data)

    #df = pd.DataFrame()
    #for i in range(1,len(data)+1):
    #    ph_df = pd.json_normalize(data[f"{i}"])
    #    df = pd.concat([df, ph_df])
    #df