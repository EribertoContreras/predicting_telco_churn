from cgi import test
import numpy as np
import seaborn as sns
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import env
from pydataset import data
import scipy
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')


def split_telco_data(df):
    '''
    This function performs split on telco data, stratify churn.
    Returns train, validate, and test dfs.
    '''
    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=123, 
                                        stratify=df.churn)
    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=123, 
                                   stratify=train_validate.churn)
    return train, validate, test

def prep_telco(df):
    #dropping duplicates
    #df.drop_duplicates(inplace=True)
    
    #drop collumns
    telco_columns_drop = ['contract_type','payment_type','internet_service_type','partner','phone_service','online_security','online_backup','device_protection','streaming_tv','streaming_movies']
    df = df.drop(columns= telco_columns_drop,axis=1)
    
    #strip spces from charges, turn to a float
    df['total_charges'] = df['total_charges'].str.strip()
    df = df[df.total_charges != '']
    df['total_charges'] = df.total_charges.astype('float')
    
    #creating numeric dummmy columns
    dummy_df = pd.get_dummies(df[['gender','dependents','multiple_lines','tech_support','paperless_billing','churn']], dummy_na=False, drop_first=[True, True])
    
    #concat
    df = pd.concat([df, dummy_df], axis = 1)
    
    #train validate test
    train, validate, test = split_telco_data(df)

    return train, validate, test
    