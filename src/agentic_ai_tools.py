import pandas as pd
import os
import psycopg2
from dotenv import load_dotenv, find_dotenv
import urllib.parse
from langchain_core.tools import tool
from thefuzz import fuzz
import re
import numpy as np
from typing import Union, List, Dict
from pydantic import BaseModel, Field

load_dotenv(find_dotenv())
DB_NAME=os.getenv('DB_NAME')
USERNAME=os.getenv('USERNAME')
PASSWORD=urllib.parse.quote(os.getenv('PASSWORD'))
HOSTNAME=os.getenv('HOSTNAME')
PORT=os.getenv('PORT')
FUZZ_RATIO_THRESHOLD = os.getenv('FUZZ_RATIO_THRESHOLD')
# Creating conneciton to database
conn = psycopg2.connect(f"dbname={DB_NAME} user={USERNAME} password={PASSWORD}")

from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional
import pandas as pd


class DataFramePayload(BaseModel):
    """
    A structured payload for safely transporting pandas DataFrames
    through LangChain tools and LLMs.

    Stores:
        - data (list of row dicts)
        - columns (list of column names)
        - index (optional)
        - dtypes (optional, for better reconstruction)
    """

    data: List[Dict[str, Any]] = Field(
        ..., description="Row-wise representation of the DataFrame."
    )
    columns: List[str] = Field(
        ..., description="Column names of the DataFrame."
    )
    index: Optional[List[Any]] = Field(
        default=None, description="Optional index values."
    )
    dtypes: Optional[Dict[str, str]] = Field(
        default=None, description="Optional dtype mapping for reconstruction."
    )

    # -------------------------------
    # Convert Payload → DataFrame
    # -------------------------------
    def to_df(self) -> pd.DataFrame:
        df = pd.DataFrame(self.data, columns=self.columns)

        # restore index if stored
        if self.index is not None:
            df.index = self.index

        # restore dtypes where possible
        if self.dtypes:
            for col, dtype in self.dtypes.items():
                try:
                    df[col] = df[col].astype(dtype)
                except Exception:
                    # silently skip any casting errors
                    pass

        return df

    # -------------------------------
    # Create Payload From DataFrame
    # -------------------------------
    @classmethod
    def from_df(cls, df: pd.DataFrame) -> "DataFramePayload":
        return cls(
            data=df.to_dict(orient="records"),
            columns=df.columns.tolist(),
            index=df.index.tolist(),
            dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
        )

    # -------------------------------
    # Validators
    # -------------------------------
    @field_validator("data")
    def validate_data(cls, v):
        if not isinstance(v, list):
            raise ValueError("data must be a list of row dicts")
        return v

    @field_validator("columns")
    def validate_columns(cls, v):
        if not isinstance(v, list):
            raise ValueError("columns must be a list of strings")
        return v



def invoice_number_match(dataframe, reference_component, threshold):
    invoice_match=False
    for idx, row in dataframe.iterrows():
        fuzzy_ratio = fuzz.ratio(str(row['invoice_number']), reference_component)
        #print(fuzzy_ratio)
        if fuzzy_ratio is not None and fuzzy_ratio>threshold:
            invoice_match = True
            print("Invoice Number Match!")
            return idx, invoice_match

def customer_number_match(dataframe, reference_component, threshold):
    customer_match=False
    for idx, row in dataframe.iterrows():
        fuzzy_ratio = fuzz.ratio(str(row['customer_number']), reference_component)
        if fuzzy_ratio is not None and fuzzy_ratio>threshold:
            customer_match=True
            print("Customer Match!")
            return idx, customer_match

def amount_number_match(dataframe, reference_component, threshold):
    is_match=0
    for idx, row in dataframe.iterrows():
        fuzzy_ratio = fuzz.ratio(str(row['amount']), reference_component)
        if fuzzy_ratio is not None and fuzzy_ratio>threshold:
            is_match=1
            print("Amount number match!")
            return idx, is_match

def fill_details(in_idx,index,payments_dataframe, accounts_receivables_dataframe):
    if accounts_receivables_dataframe.loc[in_idx,'payment'] == None:
        accounts_receivables_dataframe.loc[in_idx,'payment'] = payments_dataframe.loc[index,'payment_amount']
    else:
        accounts_receivables_dataframe.loc[in_idx,'payment'] += payments_dataframe.loc[index,'payment_amount']
    accounts_receivables_dataframe.loc[in_idx,'payment_date'] = payments_dataframe.loc[index,'payment_date']
    accounts_receivables_dataframe.loc[in_idx, 'payment_id'] = payments_dataframe.loc[index,'transaction_id']
    return accounts_receivables_dataframe

@tool
def AccessAccountsReceivable()-> DataFramePayload:
    """Function to access the accounts receivables data table in Postgres"""
    accounts_receivables = pd.read_sql("SELECT * FROM accounts_receivable", conn)
    return accounts_receivables.to_dict()

@tool
def AccessPayments():
    """Function to access the payments received data table in postgress"""
    payments = pd.read_sql("SELECT * FROM payments", conn)
    return payments.to_dict()

# @tool
# def AccessCustomer():
#     """Function to access cutomer information and payment terms"""
#     customers = pd.read_sql("SELECT * FROM customers", conn) 
#     return customers

# @tool
# def PaymentReferenceSearch(ar: dict,  pr: dict) -> dict:
#     """Fuzzy search of payment reference string for a similarity check of each string.
#     The payments_dataframe has the output of AccessPayments as input. The accounts_receivables dataframe
#     has the outut of AccessAccountsReceivable as input.

#     This function should only be used after the tools AccessAccountsReceivable and AccesssPayments have been used.

#     Function returns the updated Accounts Receivable and payments dataframe.
#     """
#     fuzz_threshold = int(FUZZ_RATIO_THRESHOLD)
#     payments_dataframe = pr.to_df()
#     accounts_receivables_dataframe = ar.to_df()
#     # Add additional column for payments_dataframe to categorise if payment has been matched or not.
#     #payments_dataframe['matched'] = False

#     try:
#         for index, row in payments_dataframe.iterrows():
#             # Try first just the payment reference information
#             print(str(row['payment_reference']))
#             pattern = r"\s"
#             string_list=re.split(pattern, str(row['payment_reference']))
#             print(string_list)
#             customer_match = False
#             invoice_match = False
#             for component in string_list:
#                 component = component.strip()
#                 print(f"Element: {component}")
#                 if component == None:
#                     pass
#                 else:
#                     # for i in range(1):
#                     #     print(f"ROUND:{i}")

#                     if customer_match:
#                         # Invoice number match
#                         print("Starting Invoice Number Match")
#                         try:
#                             in_idx, invoice_match = invoice_number_match(dataframe=accounts_receivables_dataframe,
#                                     reference_component=component,
#                                     threshold=fuzz_threshold)
                            
#                         except Exception as e:
#                             print(f"Error Invoice Match: {e}")
#                             pass
                    
#                     else:
                        
#                         print("Starting Customer Number Match")
#                         try:
#                             cs_idx, customer_match = customer_number_match(dataframe=accounts_receivables_dataframe,
#                                                 reference_component=component,
#                                                 threshold=fuzz_threshold)

#                         except Exception as e:
#                             print(f"Error Customer Match: {e}")
#                             pass
                        
#                     if invoice_match == True and customer_match == True:
#                         print("Invoice and Customer matched!")
#                         payments_dataframe.loc[index,'matched'] = True
#                         fill_details(in_idx,index, payments_dataframe,accounts_receivables_dataframe)
#                         break
                     
#                     else:
#                       payments_dataframe.loc[index,'matched'] = False
                        

                
#     except Exception as e:
#         print(f"OUTER ERROR: {e}")
#         pass

#     return DataFramePayload.from_df(accounts_receivables_dataframe)