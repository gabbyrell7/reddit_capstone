#import praw
import pandas as pd
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import networkx as nx
from collections import Counter
import creds
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

# import spacy
# import pandas as pd
# from collections import Counter

# nlp = spacy.load("en_core_web_sm")

def get_most_frequent_entity(text):
    """
    Extracts entities (ORG, GPE, PRODUCT) and returns the most frequent one.
    """
    if not isinstance(text, str) or not text:
        return None

    doc = nlp(text)
    entities = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "GPE", "PRODUCT"]]

    if not entities:
        return None

    entity_counts = Counter(entities)
    return entity_counts.most_common(1)[0][0]  # Return the most frequent entity

# Example usage (assuming 'df' is your DataFrame):

## df["most_frequent_entity"] = df["text"].apply(get_most_frequent_entity)
## df = df.dropna(subset=['most_frequent_entity']).reset_index(drop=True)
#df["entity"] = df["text"].apply(get_most_frequent_entity)
#df = df.dropna(subset=['entity']).reset_index(drop=True)
#print(df.head())

# ------------------------------------------------------------
def analyze_top_entities(df):
    """
    Counts the frequency of entities in a DataFrame column and filters the DataFrame
    to include only rows containing the top 5 most frequent entities.

    Args:
        df (pd.DataFrame): DataFrame containing a column named 'entities'
                                     (which should be a single entity per row).

    Returns:
        tuple: A tuple containing:
            - list: A list of the top 5 most frequent entities.
            - pd.DataFrame: A DataFrame filtered to include only rows where
                              the 'entities' column contains one of the top 5 entities.
    """
    # Count the frequency of each entity
    top_entities = df["entity"].value_counts().head(5).index.tolist()

    # Filter dataset to only include these top entities
    df_filtered = df[df["entity"].isin(top_entities)].reset_index(drop=True)

    print("Top 5 Entities:", top_entities)
    print(df_filtered.head())

    return top_entities, df_filtered

# Example usage (assuming 'df' is your DataFrame):

#top_entities_result, df_filtered_result = analyze_top_entities(df.copy())
# ------------------------------------------------------------

def plot_entity_frequency_over_time(df):
    """
    Plots the frequency of entities over time (created_utc).

    Args:
        df (pd.DataFrame): DataFrame with columns 'entity' and 'created_utc'.
    """

    # Convert 'created_utc' to datetime if it's not already
    df['created_utc'] = pd.to_datetime(df['created_utc'], unit='s')

    # Group by 'created_utc' and 'entity', then count occurrences
    entity_time_counts = df.groupby([pd.Grouper(key='created_utc', freq='Y'), 'entity']).size().reset_index(name='count')
    #D is daily, W Weekly, M Monthly

    # Pivot the data for plotting
    entity_time_pivot = entity_time_counts.pivot(index='created_utc', columns='entity', values='count').fillna(0)

    # Plot the data
    plt.figure(figsize=(15, 7))
    for entity in entity_time_pivot.columns:
        plt.plot(entity_time_pivot.index, entity_time_pivot[entity], label=entity)

    plt.title('Entity Frequency Over Time - {csv_name}')
    plt.xlabel('Date')
    plt.ylabel('Frequency')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
#    plt.show()
    plot_filename = os.path.join(output_dir, f'{csv_name}_entity_frequency.png')  # Save plot
    plt.savefig(plot_filename)
    plt.close()  # Close the plot to free memory

#plot_entity_frequency_over_time(df_filtered_result)

# ------------------------------------------------------------
analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    score = analyzer.polarity_scores(text)
    return score["compound"]

#df_filtered_result["sentiment"] = df_filtered_result["text"].apply(lambda x: get_sentiment(x) if isinstance(x, str) else 0)
#print(df_filtered_result.head())

# ------------------------------------------------------------
import matplotlib.pyplot as plt

def plot_entity_sentiment_over_time(df):
    """
    Plots the sentiment value of entities over time (created_utc).

    Args:
        df (pd.DataFrame): DataFrame with columns 'entity', 'created_utc', and 'sentiment'.
    """

    # Convert 'created_utc' to datetime
    df['created_utc'] = pd.to_datetime(df['created_utc'], unit='s')

    # Group by time and entity, then calculate mean sentiment
    entity_time_sentiment = df.groupby([pd.Grouper(key='created_utc', freq='D'), 'entity'])['sentiment'].mean().reset_index()

    # Pivot the data for plotting
    entity_time_pivot = entity_time_sentiment.pivot(index='created_utc', columns='entity', values='sentiment').fillna(0)

    # Plot the data
    plt.figure(figsize=(15, 7))
    for entity in entity_time_pivot.columns:
        plt.plot(entity_time_pivot.index, entity_time_pivot[entity], label=entity, marker='o', markersize=5)

    plt.title('Entity Sentiment Over Time')
    plt.xlabel('Date')
    plt.ylabel('Sentiment')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
#    plt.show()
    plot_filename = os.path.join(output_dir, f'{csv_name}_entity_sentiment.png')  # Save plot
    plt.savefig(plot_filename)
    plt.close()  # Close the plot

# Call the function
#plot_entity_sentiment_over_time(df_filtered_result)

# ------------------------------------------------------------
def process_reddit_data(csv_file_path):
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
    df.columns = ['score', 'date', 'title', 'author', 'permalink', 'text', 'id', 'created_utc', 'subreddit_id']
    print(df.head())

    # Extract entities
    df["entity"] = df["text"].apply(get_most_frequent_entity)
    df = df.dropna(subset=['entity']).reset_index(drop=True)
    print(df.head())

    # Analyze top entities
    top_entities_result, df_filtered_result = analyze_top_entities(df.copy())

    # Plot entity frequency over time
    plot_entity_frequency_over_time(df_filtered_result)

    # Perform sentiment analysis
    df_filtered_result["sentiment"] = df_filtered_result["text"].apply(
        lambda x: get_sentiment(x) if isinstance(x, str) else 0)
    print(df_filtered_result.head())

    # Plot entity sentiment over time
    plot_entity_sentiment_over_time(df_filtered_result)
    
    # Save the filtered DataFrame to a CSV
    output_csv_path = os.path.join(output_dir, f'{csv_name}_filtered.csv')
    df_filtered_result.to_csv(output_csv_path, index=False)  # Don't save the index


if __name__ == "__main__":
    # Get the input CSV file path from the command line
    if len(sys.argv) != 2:
        print("Usage: python3 your_script_name.py <input_csv_file>")
        sys.exit(1)
    csv_file_path = sys.argv[1]

    # Create an output directory
    output_dir = "OUTPUT"  # You can change this if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    process_reddit_data(csv_file_path)

