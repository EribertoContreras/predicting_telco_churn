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

def get_connection(db, user=env.user, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'
def get_titanic_data():
    return pd.read_sql('SELECT * FROM passengers', get_connection('titanic_db'))


def get_iris_data():
    return pd.read_sql('SELECT * FROM species', get_connection('iris_db'))

#def get_telco_data():
 #   return pd.read_sql("""
#SELECT * FROM customers
#JOIN customer_payments USING(customer_id)
#JOIN customer_contracts USING(customer_id)
#JOIN customer_subscriptions USING(customer_id)
#"""
#, get_connection('telco_churn'))



def get_titanic_data():
    filename = "titanic.csv"

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        # read the SQL query into a dataframe
        df = pd.read_sql('SELECT * FROM passengers', get_connection('titanic_db'))

        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename, index = False)

        # Return the dataframe to the calling code
        return df  

df = get_titanic_data()

def prep_titanic(df):
    '''
    take in titanc dataframe, remove all rows where age or embarked is null, 
    get dummy variables for sex and embark_town, 
    and drop sex, deck, passenger_id, class, and embark_town. 
    '''

    df = df[(df.age.notna()) & (df.embarked.notna())]
    df = df.drop(columns=['deck', 'passenger_id', 'class'])

    dummy_df = pd.get_dummies(df[['sex', 'embark_town']], prefix=['sex', 'embark'])

    df = pd.concat([df, dummy_df.drop(columns=['sex_male'])], axis=1)

    df = df.drop(columns=['sex', 'embark_town','embarked']) 

    df = df.rename(columns={"sex_female": "is_female"})

    return df

df = prep_titanic(df)

def train_validate_test_split(df, target, seed=123):
    '''
    This function takes in a dataframe, the name of the target variable
    (for stratification purposes), and an integer for a setting a seed
    and splits the data into train, validate and test. 
    Test is 20% of the original dataset, validate is .30*.80= 24% of the 
    original dataset, and train is .70*.80= 56% of the original dataset. 
    The function returns, in this order, train, validate and test dataframes. 
    '''
    train_validate, test = train_test_split(df, test_size=0.2, 
                                            random_state=seed, 
                                            stratify=df[target])
    train, validate = train_test_split(train_validate, test_size=0.3, 
                                       random_state=seed,
                                       stratify=train_validate[target])
    return train, validate, test

def split_data(df):
    '''
    take in a DataFrame and return train, validate, and test DataFrames; stratify on survived.
    return train, validate, test DataFrames.
    '''
    train_validate, test = train_test_split(df, test_size=.2, random_state=123, stratify=df.survived)
    train, validate = train_test_split(train_validate, 
                                       test_size=.3, 
                                       random_state=123, 
                                       stratify=train_validate.survived)
    return train, validate, test


train, validate, test = split_data(df)    

# ------------------------------------------------------------------------------------------------------------------------------
def get_iris_data():
    filename = "iris.csv"

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        # read the SQL query into a dataframe
        df = pd.read_sql('SELECT * FROM measurements JOIN species USING(species_id)', get_connection('iris_db'))

        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename, index = False)
        #changing it csv because its a csv

        # Return the dataframe to the calling code
        return df  

#train, test = train_test_split(df, test_size=.2, random_state=123, stratify=df.species_name)
#train, validate = train_test_split(train, test_size=.3, random_state=123, stratify=train.species_name)

# Validate my split.

#print(f'train -> {train.shape}')
#print(f'validate -> {validate.shape}')
#print(f'test -> {test.shape}')

def clean_iris(df):

    '''Prepares acquired Iris data for exploration'''
    
    # drop column using .drop(columns=column_name)
    df = df.drop(columns='species_id')
    
    # remame column using .rename(columns={current_column_name : replacement_column_name})
    df = df.rename(columns={'species_name':'species'})
    
    # create dummies dataframe using .get_dummies(column_name,not dropping any of the dummy columns)
    dummy_df = pd.get_dummies(df['species'], drop_first=False)
    
    # join original df with dummies df using .concat([original_df,dummy_df], join along the index)
    df = pd.concat([df, dummy_df], axis=1)
    
    return df


def split_iris_data(df):
    '''
    take in a DataFrame and return train, validate, and test DataFrames; stratify on species.
    return train, validate, test DataFrames.
    '''
    
    # splits df into train_validate and test using train_test_split() stratifying on species to get an even mix of each species
    train_validate, test = train_test_split(df, test_size=.2, random_state=123, stratify=df.species)
    
    # splits train_validate into train and validate using train_test_split() stratifying on species to get an even mix of each species
    train, validate = train_test_split(train_validate, 
                                       test_size=.3, 
                                       random_state=123, 
                                       stratify=train_validate.species)
    return train, validate, test


def prep_iris(df):
    '''Prepares acquired Iris data for exploration'''
    
    # drop column using .drop(columns=column_name)
    df = df.drop(columns='species_id')
    
    # remame column using .rename(columns={current_column_name : replacement_column_name})
    df = df.rename(columns={'species_name':'species'})
    
    # create dummies dataframe using .get_dummies(column_name,not dropping any of the dummy columns)
    dummy_df = pd.get_dummies(df['species'], drop_first=False)
    
    # join original df with dummies df using .concat([original_df,dummy_df], join along the index)
    df = pd.concat([df, dummy_df], axis=1)
    
    # split data into train/validate/test using split_data function
    train, validate, test = split_iris_data(df)
    
    return train, validate, test





def get_telco_data():
    filename = "telco.csv"

    if os.path.isfile(filename):
        return pd.read_csv(filename)

    else:
        # read the SQL query into a dataframe
        df = pd.read_sql("""
        SELECT * FROM customers
        JOIN contract_types USING(contract_type_id)
        JOIN payment_types USING(payment_type_id)
        JOIN internet_service_types USING(internet_service_type_id);""", get_connection('telco_churn'))

        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename, index = False)
        #changing it csv because its a csv

        # Return the dataframe to the calling code
        return df

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
    df.drop_duplicates(inplace=True)
    
    #drop collumns
    telco_columns_drop = ['contract_type','payment_type','internet_service_type','partner','phone_service','online_security','online_backup','device_protection','streaming_tv','streaming_movies']
    df = df.drop(columns= telco_columns_drop,axis=1)
    
    #strip spces from charges, turn to a float
    df['total_charges'] = df['total_charges'].str.strip()
    df = df[df.total_charges != '']
    df['total_charges'] = df.total_charges.astype('float')
    
    #creating numeric dummmy columns
    dummy_td = pd.get_dummies(td[['gender','dependents','multiple_lines','tech_support','paperless_billing','churn']], dummy_na=False, drop_first=[True, True])
    
    #concat
    df = pd.concat([df, dummy_td], axis = 1)
    
    #train validate test
    train, validate, test = split_telco_data(df)
    
    return train, validate, test
    



#def prep_telco_data(df):
    # Drop duplicate columns
    #df.drop(columns=['payment_type_id', 'internet_service_type_id', 'contract_type_id', 'customer_id'], inplace=True)
       
    # Drop null values stored as whitespace    
    #df['total_charges'] = df['total_charges'].str.strip()
    #df = df[df.total_charges != '']
    
    # Convert to correct datatype
    #df['total_charges'] = df.total_charges.astype(float)
    
    # Convert binary categorical variables to numeric
    #df['gender_encoded'] = df.gender.map({'Female': 1, 'Male': 0})
    #df['partner_encoded'] = df.partner.map({'Yes': 1, 'No': 0})
    #df['dependents_encoded'] = df.dependents.map({'Yes': 1, 'No': 0})
    #df['phone_service_encoded'] = df.phone_service.map({'Yes': 1, 'No': 0})
    #df['paperless_billing_encoded'] = df.paperless_billing.map({'Yes': 1, 'No': 0})
    #df['churn_encoded'] = df.churn.map({'Yes': 1, 'No': 0})
    
# Get dummies for non-binary categorical variables
    #dummy_df = pd.get_dummies(df[['multiple_lines', \
      #                        'online_security', \
     #                         'online_backup', \
     #                         'device_protection', \
    #                          'tech_support', \
     #                         'streaming_tv', \
   #                           'streaming_movies', \
 #                             'contract_type', \
  #                            'internet_service_type', \
#                              'payment_type']], dummy_na=False, \
    #                          drop_first=True)
    
    # Concatenate dummy dataframe to original 
   # df = pd.concat([df, dummy_df], axis=1)
    
    # split the data
    #train, validate, test = split_telco_data(df)
    
    #return train, validate, test
