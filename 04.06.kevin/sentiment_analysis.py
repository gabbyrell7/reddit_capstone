#import praw
import pandas as pd
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import networkx as nx
from collections import Counter
#import creds
import time
import os

#Uncomment these to have full output on jupyter
#pd.set_option('display.max_colwidth', None)
#pd.set_option('display.max_rows', None)
#pd.set_option('display.width', None)  # This one is important for terminal-like output
#pd.set_option('display.expand_frame_repr', False) # Prevents line wrapping.

# https://www.reddit.com/r/pushshift/comments/1itme1k/separate_dump_files_for_the_top_40k_subreddits/

import sys
# !{sys.executable} -m pip install spacy
# !{sys.executable} -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")


# ------------------------------------------------------------

def read_csv_to_dataframe(filepath, **kwargs):
    """
    Reads the contents of a CSV file into a Pandas DataFrame.

    Args:
        filepath (str): The path to the CSV file.
        **kwargs:  Additional keyword arguments to pass to pandas.read_csv().
                   This allows for flexibility in handling different CSV formats.

    Returns:
        pd.DataFrame: A Pandas DataFrame containing the data from the CSV file.
                      Returns an empty DataFrame if the file cannot be read.
    """
    try:
        df = pd.read_csv(filepath, **kwargs)
        return df
    except FileNotFoundError:
        print(f"Error: CSV file not found at '{filepath}'")
        return pd.DataFrame()  # Return an empty DataFrame
    except pd.errors.EmptyDataError:
        print(f"Error: CSV file at '{filepath}' is empty.")
        return pd.DataFrame()  # Return an empty DataFrame
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}")
        return pd.DataFrame()  # Return an empty DataFrame

# df = read_csv_to_dataframe("../reddit_data/reddit/2018-02-01.filtered_submissions.cut.sv")
#df = read_csv_to_dataframe("2018-03-01.filtered_submissions.csv")
#df = read_csv_to_dataframe("2018-01-01_2018-02-02.filtered_submissions/RS_2018-01.csv")
#df.columns = ['score', 'date', 'title', 'author', 'permalink','text','id','created_utc','subreddit_id']


# ------------------------------------------------------------
analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    score = analyzer.polarity_scores(text)
    return score["compound"]

#df_filtered_result["sentiment"] = df_filtered_result["text"].apply(lambda x: get_sentiment(x) if isinstance(x, str) else 0)
#print(df_filtered_result.head())

# ------------------------------------------------------------

def process_reddit_data(csv_file_path, output_dir):
    """
    Processes Reddit data from a given CSV file, performing entity extraction,
    sentiment analysis, and plotting.

    Args:
        csv_file_path (str): The path to the CSV file containing Reddit data.
    """
    # Extract the CSV filename (without extension) for use in output filenames
    csv_name = os.path.splitext(os.path.basename(csv_file_path))[0]

    # Read the CSV file into a Pandas DataFrame
    df = read_csv_to_dataframe(csv_file_path)

    # Rename the columns
    #df.columns = ['score', 'date', 'title', 'author', 'permalink', 'text', 'id', 'created_utc', 'subreddit_id', 'entity']
    print(df.head())


    # Perform sentiment analysis
    df["sentiment"] = df["text"].apply(
        lambda x: get_sentiment(x) if isinstance(x, str) else 0)
    print(df.head())

    # Save the filtered DataFrame to a CSV
    output_csv_path = os.path.join(output_dir, f'{csv_name}_all_sentiment_analysis.csv')
    df.to_csv(output_csv_path, index=False)  # Don't save the index


if __name__ == "__main__":
    # Get the input CSV file path from the command line
    if len(sys.argv) != 2:
        print("Usage: python3 your_script_name.py <input_csv_file>")
        sys.exit(1)
    csv_file_path = sys.argv[1]

    # Create an output directory
    output_dir = "OUTPUT_ALL_SENTIMENT_ANALYSIS"  # You can change this if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    process_reddit_data(csv_file_path, output_dir)

