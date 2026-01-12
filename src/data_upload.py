""" Python script to upload data to Postgres"""

# Import
import pandas as pd
import urllib.parse 
import numpy as np 
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
import psycopg2
import psycopg2.extras as extras
import time

# ENV Variables
load_dotenv()
DB_NAME=os.getenv('DB_NAME')
USERNAME=os.getenv('USERNAME')
PASSWORD=urllib.parse.quote(os.getenv('PASSWORD'))
HOSTNAME=os.getenv('HOSTNAME')
PORT=os.getenv('PORT')

conn = psycopg2.connect(
    database=DB_NAME, user=USERNAME, password=PASSWORD, host=HOSTNAME, port=PORT
    )


# Method to insert data into dataframe
def execute_values(conn, df, table):

    tuples = [tuple(x) for x in df.to_numpy()]

    cols = ','.join(list(df.columns))
    # SQL query to execute
    query = "INSERT INTO %s(%s) VALUES %%s" % (table, cols)
    cursor = conn.cursor()
    try:
        extras.execute_values(cursor, query, tuples)
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error: %s" % error)
        conn.rollback()
        cursor.close()
        return 1
    print("the dataframe is inserted")
    cursor.close()

if __name__ == "__main__":
    start = time.perf_counter()
    # import data
    #accounts_receivable
    acc_rec = pd.read_excel('transactions_upload.xlsx', sheet_name='Accounts receivable ledger')
    for col in list(acc_rec.columns):
        if "date" in col:
            acc_rec[col] = pd.to_datetime(acc_rec[col], format='mixed') 

    # Fill blanks with NaN, since blanks will be considered a string and not accepted by the postgres table
    acc_rec = acc_rec.replace({np.nan: None})


    # Customers
    customers = pd.read_excel('transactions_upload.xlsx', sheet_name='Customers')
    for col in list(customers.columns):
        if "date" in col:
            customers[col] = pd.to_datetime(customers[col], format='mixed')
    customers = customers.replace({np.nan: None})

    # Payments
    payments = pd.read_excel('transactions_upload.xlsx', sheet_name='Payments')
    for col in list(payments.columns):
        if "date" in col:
            payments[col] = pd.to_datetime(payments[col], format='mixed') 
    payments = payments.replace({np.nan: None})

    execute_values(conn, payments, 'payments')

   