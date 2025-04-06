import pandas as pd
import matplotlib.pyplot as plt
import datetime
import os
import sys

def read_csv_to_dataframe(filepath, **kwargs):
    """
    Reads the contents of a CSV file into a Pandas DataFrame.

    Args:
        filepath (str): The path to the CSV file.
        **kwargs:  Additional keyword arguments to pass to pandas.read_csv().

    Returns:
        pd.DataFrame: A Pandas DataFrame containing the data from the CSV file.
                      Returns an empty DataFrame if the file cannot be read.
    """
    try:
        df = pd.read_csv(filepath, **kwargs)
        return df
    except FileNotFoundError:
        print(f"Error: CSV file not found at '{filepath}'")
        return pd.DataFrame()
    except pd.errors.EmptyDataError:
        print(f"Error: CSV file at '{filepath}' is empty.")
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}")
        return pd.DataFrame()

def plot_entity_frequency_over_time(df, output_dir, csv_name, entities_to_plot):
    """
    Plots the frequency of entities over time (created_utc), for the specified entities.

    Args:
        df (pd.DataFrame): DataFrame with columns 'entity' and 'created_utc'.
        output_dir (str): Directory to save the plot.
        csv_name (str): Name of the input CSV file (for naming the plot).
        entities_to_plot (list): List of entities to plot.
    """

    # Convert 'created_utc' to datetime
    df['created_utc'] = pd.to_datetime(df['created_utc'], unit='s')

    # Group by time and entity, then count occurrences
    entity_time_counts = df.groupby([pd.Grouper(key='created_utc', freq='D'), 'entity']).size().reset_index(name='count')

    # Pivot the data for plotting
    entity_time_pivot = entity_time_counts.pivot(index='created_utc', columns='entity', values='count').fillna(0)

    # Filter entities to plot
    entities_to_plot = [entity for entity in entities_to_plot if entity in entity_time_pivot.columns]
    entity_time_pivot = entity_time_pivot[entities_to_plot]

    # Plot the data
    plt.figure(figsize=(15, 7))
    for entity in entity_time_pivot.columns:
        plt.plot(entity_time_pivot.index, entity_time_pivot[entity], label=entity)

    plt.title(f'Entity Frequency Over Time - {csv_name}')
    plt.xlabel('Date')
    plt.ylabel('Frequency')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_filename = os.path.join(output_dir, f'{csv_name}_entity_frequency.png')
    plt.savefig(plot_filename)
    plt.close()

def plot_entity_sentiment_over_time(df, output_dir, csv_name, entities_to_plot):
    """
    Plots the sentiment value of entities over time (created_utc), for the specified entities.

    Args:
        df (pd.DataFrame): DataFrame with columns 'entity', 'created_utc', and 'sentiment'.
        output_dir (str): Directory to save the plot.
        csv_name (str): Name of the input CSV file (for naming the plot).
        entities_to_plot (list): List of entities to plot.
    """

    # Convert 'created_utc' to datetime
    df['created_utc'] = pd.to_datetime(df['created_utc'], unit='s')

    # Group by time and entity, then calculate mean sentiment
    entity_time_sentiment = df.groupby([pd.Grouper(key='created_utc', freq='D'), 'entity'])['sentiment'].mean().reset_index()

    # Pivot the data for plotting
    entity_time_pivot = entity_time_sentiment.pivot(index='created_utc', columns='entity', values='sentiment').fillna(0)

    # Filter entities to plot
    entities_to_plot = [entity for entity in entities_to_plot if entity in entity_time_pivot.columns]
    entity_time_pivot = entity_time_pivot[entities_to_plot]

    # Plot the data
    plt.figure(figsize=(15, 7))
    for entity in entity_time_pivot.columns:
        plt.plot(entity_time_pivot.index, entity_time_pivot[entity], label=entity, marker='o', markersize=5)

    plt.title(f'Entity Sentiment Over Time - {csv_name}')
    plt.xlabel('Date')
    plt.ylabel('Sentiment')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_filename = os.path.join(output_dir, f'{csv_name}_entity_sentiment.png')
    plt.savefig(plot_filename)
    plt.close()

def process_reddit_data(csv_file_path, output_dir, entities_to_plot):
    """
    Processes Reddit data from a given CSV file, performing entity extraction,
    sentiment analysis, and plotting for specified entities.

    Args:
        csv_file_path (str): The path to the CSV file containing Reddit data.
        output_dir (str): The directory to save output files.
        entities_to_plot (list): List of entities to plot.
    """
    # Extract the CSV filename (without extension)
    csv_name = os.path.splitext(os.path.basename(csv_file_path))[0]

    # Read the CSV file into a Pandas DataFrame
    df = read_csv_to_dataframe(csv_file_path)

    # Rename the columns (ensure 'sentiment' is included)
    df.columns = ['score', 'date', 'title', 'author', 'permalink', 'text', 'id', 'created_utc', 'subreddit_id', 'entity', 'sentiment']

    # Plot entity frequency over time
    plot_entity_frequency_over_time(df, output_dir, csv_name, entities_to_plot)

    # Plot entity sentiment over time (use existing 'sentiment' column)
    plot_entity_sentiment_over_time(df.copy(), output_dir, csv_name, entities_to_plot)

if __name__ == "__main__":
    # Get the input CSV file path from the command line
    if len(sys.argv) != 2:
        print("Usage: python3 your_script_name.py <input_csv_file>")
        sys.exit(1)
    csv_file_path = sys.argv[1]

    # Define the entities to plot
    entities_to_plot = ["Reddit", "US", "RP", "YouTube", "UK", "ER", "DM", "AI", "Lunar Coin", "un", "GPA", "Google", "America"]  
    
    # Create an output directory
    output_dir = "OUTPUT_ALL_PLOTS"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    process_reddit_data(csv_file_path, output_dir, entities_to_plot)

