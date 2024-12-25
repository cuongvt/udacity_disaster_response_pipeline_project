import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    # ### 1. Load datasets.
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # ### 2. Merge datasets.
    df = pd.merge(messages, categories, on="id")

    # ### 3. Split `categories` into separate category columns.
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames

    # ### 4. Convert category values to just numbers 0 or 1.
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # ### 5. Replace `categories` column in `df` with new category columns.
    df = df.drop('categories', axis=1)
    categories.index = df.index
    frames = [df, categories]  # Or perform operations on the DFs
    return pd.concat(frames, sort=False, axis=1)

def clean_data(df):
    return df.drop_duplicates()

def save_data(df, database_filename):
    if 'sqlite:' not in database_filename:
        database_filename = 'sqlite:///' + database_filename
    engine = create_engine(database_filename)
    df.to_sql('InsertTableName', engine, index=False)


def main():
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