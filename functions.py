# important libraries
# main
import pandas as pd
import numpy as np

# additional
import unicodedata
import re

# webscraping
import requests
from bs4 import BeautifulSoup

# Machine Learning
from prophet import Prophet
import holidays
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import mean_absolute_error, mean_squared_error,  r2_score

# Sentiment analysis
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import emoji
from textblob import TextBlob
from textblob.translate import Translator
import emoji
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
from googletrans import Translator
from transformers import pipeline
import time
import random
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json

# Connection to MySQL
from dotenv import load_dotenv
import os
import pymysql
from pymysql.cursors import DictCursor
from sqlalchemy import create_engine, text


# visualization
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# hypothesis testing
from scipy import stats
import scipy.stats as st
from scipy.stats import ttest_rel
from scipy.stats import chi2_contingency
from scipy.stats.contingency import association
from statsmodels.stats.proportion import proportions_ztest # pip install statsmodels
from sklearn.preprocessing import StandardScaler

# streamlit
import streamlit as st
import pickle
from prophet.plot import plot_plotly


"""Data Cleaning"""

def data_exploration(df):
    """
    Get some insights on the data.
    param df: pandas DataFrame
    return: 
    - information about number of rows, columns, duplicates, numerical and categorical columns
    - dataframe which consists of column, data type, non-null count, missing values, and unique values. 
    """

    # check number of rows and columns
    shape = df.shape
    print("Number of rows:", shape[0])
    print("Number of columns:", shape[1])

    # check duplicates
    check_duplicates = df.duplicated().sum()
    print("Number of duplicates:", check_duplicates)

    # Create a summary DataFrame
    summary_df = pd.DataFrame({
                        'Column': df.columns,
                        'Data Type': df.dtypes,
                        'Non-Null Count': df.notnull().sum(),
                        'Missing Values': df.isnull().sum(),
                        'Unique Values': df.nunique()
                })

    # Reset index to make 'Column' a regular column
    summary_df.reset_index(drop=True, inplace=True)

    # Display the summary DataFrame
    summary_df
    
    # check numerical columns
    numerical_columns = df.select_dtypes("number").columns
    print("\nNumerical Columns:", numerical_columns)

    # check categorical columns
    categorical_columns = df.select_dtypes("object").columns
    print("\nCategorical Columns:", categorical_columns)

    return summary_df


def standardize_city_name(city):

    """
    standardize city name with character into name without character.
    return: city with standardized name
    """

    # Convert to lowercase
    city = city.lower()
    
    # Remove accents
    city = ''.join(c for c in unicodedata.normalize('NFD', city) # to normalize unicode string separating charactes with accents into base character
                   if unicodedata.category(c) != 'Mn') # check if character is not a non-spacing mark
    
    # Replace special characters with spaces
    city = re.sub(r'[^a-z0-9\s]', ' ', city) # ^ not, lowercase letter(a-z), digit(0-9), or whitespace
    
    # Replace multiple spaces with a single space
    city = re.sub(r'\s+', ' ', city)
    
    # Strip leading and trailing spaces
    city = city.strip()
    
    return city

def analyze_outliers_zscore(df, z_threshold=3):
    """
    Detect outliers using the Z-score method and provide a summary for all numeric columns.
    
    param df: pandas DataFrame
    param z_threshold: Z-score threshold for outlier detection (default 3)
    return: DataFrame with outlier summary for each numeric column
    """
    # Select numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    # Calculate Z-scores for all numeric columns
    z_scores = np.abs(stats.zscore(df[numeric_columns]))
    
    # Prepare summary data
    summary = []
    total_count = len(df)
    
    for column in numeric_columns:
        # Detect outliers
        outliers = df[z_scores[column] >= z_threshold]
        num_outliers = len(outliers)
        percentage_outliers = (num_outliers / total_count) * 100
        
        # Add to summary
        summary.append({
            'Column': column,
            'Number of Outliers': num_outliers,
            'Percentage of Outliers': percentage_outliers,
            'Lower Bound': df[column].mean() - z_threshold * df[column].std(),
            'Upper Bound': df[column].mean() + z_threshold * df[column].std(),
            'Min Outlier': outliers[column].min() if not outliers.empty else None,
            'Max Outlier': outliers[column].max() if not outliers.empty else None
        })
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary)
    
    # Sort by percentage of outliers (descending)
    summary_df = summary_df.sort_values('Percentage of Outliers', ascending=False)
    
    # Format percentage for readability
    summary_df['Percentage of Outliers'] = summary_df['Percentage of Outliers'].round(2).astype(str) + '%'
    
    return summary_df


def remove_outliers_zscore(df, z_threshold=3):
  
    """
    Remove outliers with Z-score method
    param df: pandas DataFrame
    return: DataFrame with no outliers.
    """

	# select numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns

    # Make a copy of the dataframe with only numeric columns
    df_numeric = df[numeric_columns].copy()

    # Calculate Z-scores for the specified columns
    z_scores = np.abs(stats.zscore(df_numeric))

    # Filter rows where Z-scores are below the threshold
    df_no_outliers = df[(z_scores < z_threshold).all(axis=1)]
  
    print(f"Removed {len(df) - len(df_no_outliers)} rows")
  
    return df_no_outliers


def plot_distributions(df):

    """
    Plot distribution of numerical columns.
    param df: pandas DataFrame
    return: histogram and boxplot.
    """

	# Select numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for column in numeric_columns:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        sns.histplot(df[column], kde=True, ax=ax[0])
        sns.boxplot(x=df[column], ax=ax[1])
        ax[0].set_title(f'Distribution of {column}')
        ax[1].set_title(f'Boxplot of {column}')
        plt.show()


def euclidian_distance(x1,x2,y1,y2):
    """"To calculate distance between two points with input of their latitude and longitude"""

    d  = round((((abs(x2) - abs(x1))**2 + (y2-y1)**2)**0.5),2)
    
    return d


def calculate_duration(row):
    """Calculate the delivery duration"""

    if row['status'] == 'delivered':
        if pd.notna(row['delivered_carrier_timestamp']) and pd.notna(row['delivered_customer_timestamp']):
            duration = (row['delivered_customer_timestamp'] - row['delivered_carrier_timestamp']).days
            return duration
    return 0


"""EDA"""


def univariate_numerical(df):
    """
    Plot distribution of numerical columns.
    param df: pandas DataFrame
    return: histogram and boxplot side by side for each numerical column, and summary statistics.
    """

    df_num = df.select_dtypes(include=[np.number])
    
    # Calculate statistics
    summary_stats = []
    for col in df_num.columns:
        stats = {
            'Column': col,
            'Mean': round(df_num[col].mean(), 2),
            'Median': round(df_num[col].median(), 2),
            'Mode': round(df_num[col].mode().iloc[0], 2),
            'Variance': round(df_num[col].var(), 2),
            'Standard Deviation': round(df_num[col].std(), 2),
            'Min Value': df_num[col].min(),
            'Max Value': df_num[col].max(),
            'Range': df_num[col].max() - df_num[col].min(),
            'Interquartile Range': df_num[col].quantile(0.75) - df_num[col].quantile(0.25),
            'Skewness': round(df_num[col].skew(), 2),
            'Kurtosis': round(df_num[col].kurtosis(), 2)
        }
        summary_stats.append(stats)

    summary_df = pd.DataFrame(summary_stats)

    # Visualization
    for col in df_num.columns:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))  # Side by side, reduced size

        # Histogram plot
        sns.histplot(data=df_num, x=col, ax=ax1, kde=True, bins=20, color="skyblue")
        ax1.set_title(f'Histogram of {col}')

        # Box plot
        sns.boxplot(data=df_num, x=col, ax=ax2, color="skyblue")
        ax2.set_title(f'Box Plot of {col}')

        plt.tight_layout()
        plt.show()
    
    return summary_df


def summary_categorical_correlation(df, target_column):
    """Summarize categorical correlation of a dataframe"""
    categorical_columns = df.select_dtypes(['object', 'category']).columns
    categorical_columns = [col for col in categorical_columns if col != target_column]

    results = []

    for column in categorical_columns:
        crosstab_result = pd.crosstab(df[column], df[target_column])

        # Chi-square test
        chi2_statistic, chi2_p_value, _, _ = chi2_contingency(crosstab_result)

        # Cramer's V
        cramer = association(crosstab_result, method="cramer")

        # Append all results
        results.append({
            'Column': column,
            'Chi2 p-value': chi2_p_value,
            'Cramer V': cramer
        })
    
    # Create a DataFrame for results
    results_df = pd.DataFrame(results)

    return results_df


"""Import Data to MySQL Database"""

def insert_filtered_customers(df_customers, batch_size=1000):
    """To import customers data into SQL because not all data matches the geolocation data which may result in error"""
    
    # Create a SQLAlchemy engine
    engine = create_engine(f"mysql+pymysql://root:{os.getenv('mySQL_password')}@127.0.0.1/mydb")

    # Fetch all unique_index values from geolocation table
    with engine.connect() as connection:
        result = connection.execute(text("SELECT DISTINCT unique_index FROM geolocation"))
        valid_unique_indices = {row[0] for row in result}

    # Filter the customers DataFrame
    df_customers_filtered = df_customers[df_customers['unique_index'].isin(valid_unique_indices)]

    # Check how many records were filtered out
    filtered_out_count = len(df_customers) - len(df_customers_filtered)
    print(f"Filtered out {filtered_out_count} records due to missing geolocation data.")

    # Convert filtered DataFrame to list of dictionaries
    customers_data = df_customers_filtered.to_dict('records')

    connection = pymysql.connect(host='127.0.0.1',
                                 user='root',
                                 password=os.getenv('mySQL_password'),
                                 database='mydb',
                                 cursorclass=DictCursor)
    try:
        with connection.cursor() as cursor:
            for i in range(0, len(customers_data), batch_size):
                batch = customers_data[i:i+batch_size]
                try:
                    sql = """INSERT IGNORE INTO customers 
                             (customer_id, unique_id, zip_code_prefix, city, state_code, unique_index, customer_unique_id) 
                             VALUES (%(customer_id)s, %(unique_id)s, %(zip_code_prefix)s, %(city)s, 
                                     %(state_code)s, %(unique_index)s, %(customer_unique_id)s)"""
                    cursor.executemany(sql, batch)
                    connection.commit()
                    # print(f"Inserted batch {i//batch_size + 1}")
                except pymysql.Error as e:
                    print(f"Error in batch {i//batch_size + 1}: {e}")
                    for record in batch:
                        try:
                            cursor.execute(sql, record)
                            connection.commit()
                        except pymysql.Error as e:
                            print(f"Error with record: {record}")
                            print(f"Error message: {e}")
                    connection.rollback()
    except pymysql.Error as e:
        print(f"Database error: {e}")
    finally:
        connection.close()

    print("Data insertion completed.")

    # Check for duplicate keys in the original dataframe
    duplicates = df_customers[df_customers.duplicated(subset=['customer_id'], keep=False)]
    if not duplicates.empty:
        print("Duplicate customer_ids found in original data:")
        print(duplicates)

    # Save filtered out records
    filtered_out_records = df_customers[~df_customers['unique_index'].isin(valid_unique_indices)]
    if not filtered_out_records.empty:
        filtered_out_records.to_csv('../../data/cleaned/filtered_out_customers.csv', index=False)
        print(f"Saved {len(filtered_out_records)} filtered out records to 'filtered_out_customers.csv'")


def insert_filtered_sellers(df_sellers, batch_size=1000):
    """To import sellers data into SQL because not all data matches the geolocation data which may result in error"""
    
    # Create a SQLAlchemy engine
    engine = create_engine(f"mysql+pymysql://root:{os.getenv('mySQL_password')}@127.0.0.1/mydb")

    # Fetch all unique_index values from geolocation table
    with engine.connect() as connection:
        result = connection.execute(text("SELECT DISTINCT unique_index FROM geolocation"))
        valid_unique_indices = {row[0] for row in result}

    # Filter the customers DataFrame
    df_filtered = df_sellers[df_sellers['unique_index'].isin(valid_unique_indices)]

    # Check how many records were filtered out
    filtered_out_count = len(df_sellers) - len(df_filtered)
    print(f"Filtered out {filtered_out_count} records due to missing geolocation data.")

    # Convert filtered DataFrame to list of dictionaries
    sellers_data = df_filtered.to_dict('records')

    connection = pymysql.connect(host='127.0.0.1',
                                 user='root',
                                 password=os.getenv('mySQL_password'),
                                 database='mydb',
                                 cursorclass=DictCursor)
    try:
        with connection.cursor() as cursor:
            for i in range(0, len(sellers_data), batch_size):
                batch = sellers_data[i:i+batch_size]
                try:
                    sql = """INSERT IGNORE INTO sellers
                             (seller_id, zip_code_prefix, city, state_code, unique_index) 
                             VALUES (%(seller_id)s, %(zip_code_prefix)s, %(city)s, 
                                     %(state_code)s, %(unique_index)s)"""
                    cursor.executemany(sql, batch)
                    connection.commit()
                    # print(f"Inserted batch {i//batch_size + 1}")
                except pymysql.Error as e:
                    print(f"Error in batch {i//batch_size + 1}: {e}")
                    for record in batch:
                        try:
                            cursor.execute(sql, record)
                            connection.commit()
                        except pymysql.Error as e:
                            print(f"Error with record: {record}")
                            print(f"Error message: {e}")
                    connection.rollback()
    except pymysql.Error as e:
        print(f"Database error: {e}")
    finally:
        connection.close()

    print("Data insertion completed.")

    # Check for duplicate keys in the original dataframe
    duplicates = df_sellers[df_sellers.duplicated(subset=['seller_id'], keep=False)]
    if not duplicates.empty:
        print("Duplicate sellers_ids found in original data:")
        print(duplicates)

    # Save filtered out records
    filtered_out_records = df_sellers[~df_sellers['unique_index'].isin(valid_unique_indices)]
    if not filtered_out_records.empty:
        filtered_out_records.to_csv('../../data/cleaned/filtered_out_sellers.csv', index=False)
        print(f"Saved {len(filtered_out_records)} filtered out records to 'filtered_out_sellers.csv'")


def insert_filtered_items(df_items, batch_size=1000):
    """To import items data into SQL because not all data matches with reference tables"""
    
    # Create a SQLAlchemy engine
    engine = create_engine(f"mysql+pymysql://root:{os.getenv('mySQL_password')}@127.0.0.1/mydb")

    # Fetch all valid order_ids, product_ids, and seller_ids from their respective tables
    with engine.connect() as connection:
        valid_order_ids = set(row[0] for row in connection.execute(text("SELECT order_id FROM orders")))
        valid_product_ids = set(row[0] for row in connection.execute(text("SELECT product_id FROM products")))
        valid_seller_ids = set(row[0] for row in connection.execute(text("SELECT seller_id FROM sellers")))

    print(f"Number of valid order_ids: {len(valid_order_ids)}")
    print(f"Number of valid product_ids: {len(valid_product_ids)}")
    print(f"Number of valid seller_ids: {len(valid_seller_ids)}")

    # Filter the items DataFrame
    df_filtered = df_items[
        df_items['order_id'].isin(valid_order_ids) &
        df_items['product_id'].isin(valid_product_ids) &
        df_items['seller_id'].isin(valid_seller_ids)
    ]

    # Check how many records were filtered out
    filtered_out_count = len(df_items) - len(df_filtered)
    print(f"Filtered out {filtered_out_count} records due to missing foreign key references.")

    # Debugging: Check which foreign keys are causing the filtering
    order_id_mismatch = df_items[~df_items['order_id'].isin(valid_order_ids)]
    product_id_mismatch = df_items[~df_items['product_id'].isin(valid_product_ids)]
    seller_id_mismatch = df_items[~df_items['seller_id'].isin(valid_seller_ids)]

    print(f"Records with invalid order_id: {len(order_id_mismatch)}")
    print(f"Records with invalid product_id: {len(product_id_mismatch)}")
    print(f"Records with invalid seller_id: {len(seller_id_mismatch)}")

    # Print a few examples of mismatched IDs
    print("\nExample mismatched order_ids:")
    print(order_id_mismatch['order_id'].head())
    print("\nExample mismatched product_ids:")
    print(product_id_mismatch['product_id'].head())
    print("\nExample mismatched seller_ids:")
    print(seller_id_mismatch['seller_id'].head())

    # Check data types
    print("\nData types in df_items:")
    print(df_items.dtypes)
    print("\nData type of valid_order_ids:", type(next(iter(valid_order_ids))))
    print("Data type of valid_product_ids:", type(next(iter(valid_product_ids))))
    print("Data type of valid_seller_ids:", type(next(iter(valid_seller_ids))))

    
    # Convert filtered DataFrame to list of dictionaries
    items_data = df_filtered.to_dict('records')

    connection = pymysql.connect(host='127.0.0.1',
                                 user='root',
                                 password=os.getenv('mySQL_password'),
                                 database='mydb',
                                 cursorclass=DictCursor)
    try:
        with connection.cursor() as cursor:
            for i in range(0, len(items_data), batch_size):
                batch = items_data[i:i+batch_size]
                try:
                    sql = """INSERT IGNORE INTO items
                             (order_id, item_id, product_id, seller_id, shipping_limit_timestamp, price, freight_value) 
                             VALUES (%(order_id)s, %(item_id)s, %(product_id)s, %(seller_id)s, 
                                     %(shipping_limit_timestamp)s, %(price)s, %(freight_value)s)"""
                    cursor.executemany(sql, batch)
                    connection.commit()
                    # print(f"Inserted batch {i//batch_size + 1}")
                except pymysql.Error as e:
                    print(f"Error in batch {i//batch_size + 1}: {e}")
                    for record in batch:
                        try:
                            cursor.execute(sql, record)
                            connection.commit()
                        except pymysql.Error as e:
                            print(f"Error with record: {record}")
                            print(f"Error message: {e}")
                    connection.rollback()
    except pymysql.Error as e:
        print(f"Database error: {e}")
    finally:
        connection.close()

    print("Data insertion completed.")

    # Check for duplicate keys in the original dataframe
    duplicates = df_items[df_items.duplicated(subset=['order_id', 'item_id'], keep=False)]
    if not duplicates.empty:
        print("Duplicate primary keys (order_id, item_id) found in original data:")
        print(duplicates)
        duplicates.to_csv('../../data/cleaned/duplicate_items.csv', index=False)
        print(f"Saved {len(duplicates)} duplicate records to 'duplicate_items.csv'")

    # Save filtered out records
    filtered_out_records = df_items[
        ~(df_items['order_id'].isin(valid_order_ids) &
          df_items['product_id'].isin(valid_product_ids) &
          df_items['seller_id'].isin(valid_seller_ids))
    ]
    if not filtered_out_records.empty:
        filtered_out_records.to_csv('../../data/cleaned/filtered_out_items.csv', index=False)
        print(f"Saved {len(filtered_out_records)} filtered out records to 'filtered_out_items.csv'")




"""Hypothesis Testing"""

def normality_check(df, column_name):
    """To check the ditribution whether it follows normal distribution or not"""

    column = df[column_name]

    # Histogram plot to understand the distribution of data
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(column, kde=True, bins=30, color="salmon")
    plt.title(f'Histogram of {column_name}')

    # Q-Q plot
    plt.subplot(1, 2, 2)
    stats.probplot(column, dist="norm", plot=plt)
    stats.probplot(column, plot=plt)
    plt.title(f'Q-Q Plot of {column_name}')

    plt.tight_layout()
    plt.show()

    # Conducting the Kolmogorov-Smirnov test
    standardized_column = (column - column.mean()) / column.std()
    ks_test_statistic, ks_p_value = stats.kstest(standardized_column, 'norm')

    ks_test_statistic, ks_p_value

    # print the test result
    print('The p value is', ks_p_value)

    if ks_p_value < 0.05:
        print('The test results indicate that the distribution is significantly different from a normal distribution.')
    else:
        print('The test results indicate that the distribution is not significantly different from a normal distribution.')

def data_normalization(df, column_name):
    """To normalize data which doesn't follow normal distribution"""

    # transform the data
    log_transformed_column = np.log1p(df[column_name])
    standardized_log_column = (
        log_transformed_column - log_transformed_column.mean()) / log_transformed_column.std()

    # Plotting histogram for transformed 'column_name'
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(standardized_log_column, kde=True, bins=50, color="skyblue")
    plt.title(f'Histogram of normalized {column_name}')

    # Q-Q plot
    plt.subplot(1, 2, 2)
    stats.probplot(standardized_log_column, plot=plt)
    plt.title(f'Q-Q Plot of normalized {column_name}')

    plt.tight_layout()
    plt.show()

    # Conducting the Kolmogorov-Smirnov test on the log-transformed and standardized column
    ks_test_statistic_after_transformation, ks_p_value_after_transformation = stats.kstest(
        standardized_log_column, 'norm')

    ks_test_statistic_after_transformation, ks_p_value_after_transformation

    # update the standardized column
    scaler = StandardScaler()
    log_transformed_standardized = scaler.fit_transform(
        # standardize log_transformed_column
        log_transformed_column.values.reshape(-1, 1))

    df[column_name] = scaler.inverse_transform(log_transformed_standardized)


def normality_check_group(df, column_name, group_column):
    """To check normality of a grouped columns"""
    
    unique_groups = df[group_column].unique()
    for group in unique_groups:
        group_data = df[df[group_column] == group][column_name]
        
        # Histogram and Q-Q plot
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.histplot(group_data, kde=True, bins=30, color="salmon")
        plt.title(f'Histogram of {column_name} in {group}')
        
        plt.subplot(1, 2, 2)
        stats.probplot(group_data, dist="norm", plot=plt)
        plt.title(f'Q-Q Plot of {column_name} in {group}')
        
        plt.tight_layout()
        plt.show()
        
        # Kolmogorov-Smirnov test
        standardized_group_data = (group_data - group_data.mean()) / group_data.std()
        ks_test_statistic, ks_p_value = stats.kstest(standardized_group_data, 'norm')
        
        print(f'Kolmogorov-Smirnov test for {group}:')
        print(f'The p-value is {ks_p_value}')
        
        if ks_p_value < 0.05:
            print(f'The distribution of {column_name} in {group} is significantly different from normal.\n')
        else:
            print(f'The distribution of {column_name} in {group} is not significantly different from normal.\n')


def data_normalization_group(df, column_name, group_column):
    """To normalize data of a group columns"""
    
    unique_groups = df[group_column].unique()
    for group in unique_groups:
        group_data = df[df[group_column] == group][column_name]
        
        # Log transform and standardize
        log_transformed_column = np.log1p(group_data)
        standardized_log_column = (log_transformed_column - log_transformed_column.mean()) / log_transformed_column.std()
        
        # Histogram and Q-Q plot for transformed data
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.histplot(standardized_log_column, kde=True, bins=50, color="skyblue")
        plt.title(f'Histogram of normalized {column_name} in {group}')
        
        plt.subplot(1, 2, 2)
        stats.probplot(standardized_log_column, plot=plt)
        plt.title(f'Q-Q Plot of normalized {column_name} in {group}')
        
        plt.tight_layout()
        plt.show()
        
        # Kolmogorov-Smirnov test for transformed data
        ks_test_statistic, ks_p_value = stats.kstest(standardized_log_column, 'norm')
        
        print(f'Kolmogorov-Smirnov test after transformation for {group}:')
        print(f'The p-value is {ks_p_value}')
        
        # Updating the dataframe with transformed data (optional)
        df.loc[df[group_column] == group, column_name] = standardized_log_column

    return df


"""Machine Learning"""

def cyclical_encoding(data, column, max_value):
    """To create sine cosine for time series machine learning"""
    
    data[f'{column}_sin'] = np.sin(2 * np.pi * data[column]/max_value)
    data[f'{column}_cos'] = np.cos(2 * np.pi * data[column]/max_value)
    return data

def train_and_evaluate_models(X_train_scaled, X_test_scaled, y_train, y_test, random_state=42):
    """To train and evaluate the models"""
    
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=random_state),
        'XGBoost': xgb.XGBRegressor(n_estimators=200, random_state=random_state),
        'CatBoost': CatBoostRegressor(iterations=1000, learning_rate=0.1, depth=6, verbose=100)
    }

    results = []

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results.append({
            'Model': name,
            'Mean Squared Error': mse,
            'Root Mean Squared Error': rmse,
            'Mean Absolute Error': mae,
            'R-squared Score': r2
        })

    # Create DataFrame
    performance_df = pd.DataFrame(results)
    performance_df = performance_df.set_index('Model')
    
    # Display all rows and sort by R-squared Score in descending order
    pd.set_option('display.max_rows', None)
    performance_df.sort_values(by = "R-squared Score", ascending = False)

    return performance_df

def hyperparameter_tuning(X_train_scaled, X_test_scaled, y_train, y_test, previous_model_results, n_top_models=3, cv=3, random_state=42):
    """"To perform hyperparameter tuning finding best models and parameters"""
    
    # Select top N models from previous results
    top_models = previous_model_results.nlargest(n_top_models, 'R-squared Score')
    
    param_grids = {
        'Random Forest': {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        },
        'XGBoost': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 4, 5]
        },
        'CatBoost': {
            'iterations': [500, 1000],    # control the number of trees. More iterations can lead ot better performance, but may cause overfitting
            'learning_rate': [0.05, 0.1],
            'depth': [6, 8],
            'l2_leaf_reg': [3, 5, 7],     # could reduce overfitting
            'border_count': [32, 64]       # controls the granulity of feature discretization. Affect performance and training speed
        }
    }

    models = {
        'Random Forest': RandomForestRegressor(random_state=random_state),
        'XGBoost': xgb.XGBRegressor(random_state=random_state),
        'CatBoost': CatBoostRegressor(random_state=random_state)
    }

    results = []

    for model_name in top_models.index:
        if model_name not in param_grids:
            print(f"No hyperparameter grid defined for {model_name}. Skipping.")
            continue

        print(f"Tuning {model_name}...")
        model = models[model_name]
        param_grid = param_grids[model_name]

        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, n_jobs=-1, verbose=1, scoring='r2')
        grid_search.fit(X_train_scaled, y_train)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test_scaled)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results.append({
            'Model': model_name,
            'Best Parameters': grid_search.best_params_,
            'Mean Squared Error': mse,
            'Root Mean Squared Error': rmse,
            'Mean Absolute Error': mae,
            'R-squared Score': r2
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.set_index('Model')
    
    # Set display options
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', 1000)

    results_df = results_df.sort_values('R-squared Score', ascending=False)

    return results_df


""""Sentiment Analysis"""

def extract_emojis(text):
    """Function to extract emojis"""
    return ''.join(c for c in text if emoji.is_emoji(c))

def setup_nltk():
    """To setup NLTK"""
    nltk.download('stopwords')
    nltk.download('punkt_tab')

def get_portuguese_stopwords():
    return set(stopwords.words('portuguese'))


def clean_text(text, stopwords):
    """Function to clean text"""
    # Extract emojis
    emojis = extract_emojis(text)
    
    # Remove emojis and other special characters, keeping Portuguese letters
    text = re.sub(r'[^\w\s\dáàâãéèêíïóôõöúçñÁÀÂÃÉÈÍÏÓÔÕÖÚÇÑ]', '', text)
    
    # Convert to lowercase and tokenize
    tokens = word_tokenize(text.lower())
    
    # Remove stopwords
    cleaned_tokens = [token for token in tokens if token not in stopwords]
    
    return ' '.join(cleaned_tokens), emojis


def is_valid_word(word):
    """Function to filter out non-alphabetic tokens"""
    return re.match(r'^[a-zA-ZáàâãéèêíïóôõöúçñÁÀÂÃÉÈÍÏÓÔÕÖÚÇÑ]+$', word) is not None


def translate_to_english(word):
    """Function to translate a word from Portuguese to English"""
    translator = Translator()
    try:
        return translator.translate(word, src='pt', dest='en').text.lower()
    except Exception as e:
        print(f"Error translating {word}: {e}")
        return word


def get_sentiment(text):
    """Translate to English (TextBlob's sentiment analysis works best for English)"""
    translator = Translator()
    translated = translator.translate(text, from_lang='pt', to_lang='en')
    return TextBlob(str(translated)).sentiment.polarity

