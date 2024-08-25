import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    This function reads the paths of two datasets and merges them.
    
    Input:
    Path of two datasets
    
    Output:
    The merge of the two loaded datasets
    
    '''
    df_categories = pd.read_csv(categories_filepath)
    df_messages = pd.read_csv(messages_filepath)
    df_merged = pd.merge(df_messages, df_categories, on="id",how="left")
    return df_merged 


def clean_data(df):
    '''
    The function splits the concatenated category data into individual columns, cleans and converts the values 
    from strings to integers, and then merges these columns back into the original DataFrame. It also removes 
    duplicate rows from the DataFrame to ensure the data is unique.
    
    Input:
    Pandas dataframe
    
    Output:
    Cleaned dataframe
    
    '''
    categories = df['categories'].str.split(';', expand = True)
    row = categories.iloc[0].str.split('-', expand=True)
    category_colnames = row[0]
    categories.columns = category_colnames
    
    for column in categories:
        categories[column] = categories[column].str[-1]
        
        categories[column] = pd.to_numeric(categories[column])
        
    df = df.drop('categories', axis=1)
    df = pd.concat([df,categories], axis = 1)
    df = df.drop_duplicates()
    return df


def save_data(df, database_filename):
    '''
    This function takes a DataFrame and a database filename as input. It connects to the SQLite database using SQLAlchemy, 
    then saves the DataFrame to a table named 'Messages'. The function uses the 'replace' option to ensure that the table 
    is overwritten if it already exists.
    
    Input:
    Pandas dataframe
    
    Output:
    The function doesn't return anything. It saves the DataFrame to the specified SQLite database file.
    
    '''
    engine = create_engine('sqlite:///' + str(database_filename), echo=False)
    df.to_sql('Messages', engine, if_exists = 'replace',index=False)
    pass  


def main():
    '''
    The function loads data from two CSV files (messages and categories), cleans the data, 
    and then saves the cleaned data to an SQLite database. If the user does not provide the 
    correct number of arguments, the function prints a usage message with instructions.
    
    Input: 
        Command-line arguments:
            1. messages_filepath:
                - The file path of the CSV file containing the disaster messages.
        
            2. categories_filepath:
                - The file path of the CSV file containing the categories associated with the messages.
        
            3. database_filepath:
                - The file path where the cleaned data should be saved in an SQLite database.
    
    Output:
    The function does not return any values. It processes the input data and saves the cleaned DataFrame to a database.
    
    
    '''
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()