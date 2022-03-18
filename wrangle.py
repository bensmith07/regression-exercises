import os
import pandas as pd
import env
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer

def get_zillow():
    
    filename = 'zillow.csv'
    
    if os.path.exists(filename):
        print('Reading from local CSV...')
        return pd.read_csv(filename)
    
    url = env.get_db_url('zillow')
    sql = '''
            SELECT bedroomcnt, 
                   bathroomcnt, 
                   calculatedfinishedsquarefeet, 
                   taxvaluedollarcnt, 
                   yearbuilt, 
                   taxamount,
                   fips 
              FROM properties_2016
                LEFT JOIN propertylandusetype USING (propertylandusetypeid)
              WHERE propertylandusedesc IN ("Single Family Residential", 
                                            "Inferred Single Family Residential");
            '''
    
    print('No local file exists\nReading from SQL database...')
    df = pd.read_sql(sql, url)

    print('Saving to local CSV... ')
    df.to_csv(filename, index=False)
    
    return df

def prep_zillow_1(df):
    
    # check for null values
    total_nulls = df.isnull().sum().sum()
    
    # if the total number of nulls is less than 5% of the number of observations in the df
    if total_nulls / len(df) < .05:
        # drop all rows containing null values
        df = df.dropna()
    else:
        print('Number of null values > 5% length of df. Evaluate further before dropping nulls.')
    
    # renaming columns for readability
    df = df.rename(columns = {'bedroomcnt': 'bedrooms',
                              'bathroomcnt': 'bathrooms', 
                              'calculatedfinishedsquarefeet': 'sqft', 
                              'taxvaluedollarcnt': 'tax_value',
                              'yearbuilt': 'year_built',
                              'taxamount': 'tax_amount'})
    # changing data types
    
    # changing year from float to int
    df['year_built'] = df.year_built.apply(lambda year: int(year))
    
    # changing fips codes to strings
    df['fips'] = df.fips.apply(lambda fips: '0' + str(int(fips)))
    
    return df

def wrangle_zillow():
    df = get_zillow()
    df = prep_zillow_1(df)
    return df

def train_test_validate_split(df, test_size=.2, validate_size=.3, random_state=42):
    '''
    This function takes in a dataframe, then splits that dataframe into three separate samples
    called train, test, and validate, for use in machine learning modeling.

    Three dataframes are returned in the following order: train, test, validate. 
    
    The function also prints the size of each sample.
    '''
    train, test = train_test_split(df, test_size=.2, random_state=42)
    train, validate = train_test_split(train, test_size=.3, random_state=42)
    
    print(f'train\t n = {train.shape[0]}')
    print(f'test\t n = {test.shape[0]}')
    print(f'validate n = {validate.shape[0]}')
    
    return train, test, validate

def scale_zillow(train, validate, test, scaler_type=MinMaxScaler()):    
    features_to_scale = ['bedrooms', 'bathrooms', 'sqft', 'year_built', 'tax_amount']
    other_features = ['fips']
    target = 'tax_value'

    # establish empty dataframes for storing scaled dataset
    train_scaled = pd.DataFrame(index=train.index)
    validate_scaled = pd.DataFrame(index=validate.index)
    test_scaled = pd.DataFrame(index=test.index)

    # screate and fit the scaler
    scaler = scaler_type.fit(train[features_to_scale])

    # adding scaled features to scaled dataframes
    train_scaled[features_to_scale] = scaler.transform(train[features_to_scale])
    validate_scaled[features_to_scale] = scaler.transform(validate[features_to_scale])
    test_scaled[features_to_scale] = scaler.transform(test[features_to_scale])

    # adding other features (no scaling) to scaled dataframes
    train_scaled[other_features] = train[other_features]
    validate_scaled[other_features] = validate[other_features]
    test_scaled[other_features] = test[other_features]

    # adding target variable (no scaling) to scaled dataframes
    train_scaled[target] = train[target]
    validate_scaled[target] = validate[target]
    test_scaled[target] = test[target]
    
    return train_scaled, validate_scaled, test_scaled

def prep_telco_1(df):
    '''
    This function returns a cleaned and prepped version of the Telco Customer DataFrame by accomplishing the following: 
    - drops any duplicate rows that may be present, so as not to count a customer twice
    - fixes a formatting issue in the total_charges column by replacing white space characters with nulls
    - removes customers with a 0 value for tenure (brand new customers who have not had an opportunity to churn are not relevant to our study)
        - note: this also removed rows with missing values in the total_charges column (there were no other missing values in the dataset)
    - drops unnecessary or unhelpful columns which can provide no additional predictive value, including:
        - payment_type_id
        - internet_service_type_id
        - contract_type id
        - customer_id
        - total_charges (because it is merely a function of monthly charges and tenure)
    - changes values in the senior_citizen column to Yes/No instead of 1/0 for readability
    - creates new features, including: 
        - tenure_quarters, which represents which quarter of service a customer is currently in (or was in at the time of churn). 
        - tenure_years, which represents which year of service a customer is currently in (or was in at the time of churn). 
    '''

    # drop duplicate rows, if present
    df = df.drop_duplicates()

    # clean up total_charges column and cast as float
    df['total_charges'] = df.total_charges.replace(' ', np.nan).astype(float)

    # drop rows with any null values
    df = df.dropna()

    # removing brand new customers
    df = df[df.tenure != 0]

    # drop any unnecessary, unhelpful, or duplicated columns. 
    # type_id columns are simply foreign key columns that have corresponding string values
    # customer_id is a primary key that is not useful for our analysis
    # total_charges is essentially a function of monthly_charges * tenure
    df = df.drop(columns=['payment_type_id', 'internet_service_type_id', 'contract_type_id'])

    # change senior citizen to object types Yes/No for exploration purposes
    df['senior_citizen'] = df.senior_citizen.map({1: 'Yes', 0: 'No'})

    # rename tenure column
    df = df.rename(columns={'tenure': 'tenure_months'})

    return df


#### prep 2

def prep_telco_2(df):
    '''
    This function takes in the Telco Customer dataframe, defines categorical columns as any column with an object data type 
    (except for customer_id), then adds encoded versions of  those columns for machine learning using one-hot encoding via the pandas 
    get_dummies function. Then returns the resulting dataframe. 
    '''
    # define categorical columns
    categorical_columns = list(df.dtypes[df.dtypes == 'object'].drop(columns='customer_id').index)
    categorical_columns.remove('customer_id')

    # one-hot encoding those columns
    for col in categorical_columns:
        dummy_df = pd.get_dummies(df[col],
                                prefix=f'enc_{df[col].name}',
                                drop_first=True,
                                dummy_na=False)
        
        # add the columns to the dataframe
        df = pd.concat([df, dummy_df], axis=1)
        
    # clean up the column names
    for col in df.columns:
        df = df.rename(columns={col: col.lower()})
        df = df.rename(columns={col: col.replace(' ', '_')})
    df = df.rename(columns={'enc_churn_yes': 'enc_churn'})

    return df

# define a function for obtaining Telco Customer Data
def get_telco_data():
    '''
    This function acquires TelCo customer data from the Codeup MySQL database, or from a .csv 
    file in the local directory. If a local .csv exists, the .csv is imported using pandas. If
    there is no local file, data.codeup.com is accessed with the appropriate credentials, and the 
    data is obtained via SQL query and then imported via pandas. The SQL query joins all necessary
    tables for customer data. After obtaining from the database, the pandas dataframe is cached to 
    a local CSV for future use. The data is returned from the function as a pandas dataframe. 
    '''
    
    filename = 'telco_churn.csv'
    
    # check for existing csv file in local directory
    # if it exists, return it as a datframe
    if os.path.exists(filename):
        print('Reading from local CSV...')
        return pd.read_csv(filename)
    
    # if no local directory exists, query the codeup SQL database 
    
    # utilize function defined in env.py to define the url
    url = env.get_db_url('telco_churn')
    
    # join the customer, contract_types, internet_service_types, and payment_types tables
    sql = '''
    SELECT * 
      FROM customers
        JOIN contract_types USING(contract_type_id)
        JOIN internet_service_types USING(internet_service_type_id)
        JOIN payment_types USING(payment_type_id)
    '''
    
    # return  the results of the query as a dataframe
    print('No local file exists\nReading from SQL database...')
    df = pd.read_sql(sql, url)
    
    # save the dataframe to the local directory as a CSV for future ease of access
    print('Saving to local CSV...')
    df.to_csv(filename, index=False)
    
    # return the dataframe
    return df