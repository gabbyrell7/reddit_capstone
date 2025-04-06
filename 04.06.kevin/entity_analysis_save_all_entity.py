import pandas as pd
import spacy
import os
import sys
from collections import Counter

nlp = spacy.load("en_core_web_sm")

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

def get_most_frequent_entity(text):
    """
    Extracts entities (ORG, GPE, PRODUCT) from a text and returns the most frequent one.
    Handles potential errors during spaCy processing.
    """
    if not isinstance(text, str) or not text:
        return None

    try:
        doc = nlp(text)
        entities = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "GPE", "PRODUCT"]]
        if not entities:
            return None
        entity_counts = Counter(entities)
        return entity_counts.most_common(1)[0][0]
    except Exception as e:
        print(f"Error during entity extraction: {e}")
        return None  # Or handle the error as appropriate for your use case

def process_reddit_data(csv_file_path, output_dir, batch_size=1000):
    """
    Processes Reddit data from a given CSV file, performing entity extraction
    using spaCy's pipe, and saves a CSV with all posts and their entities.

    Args:
        csv_file_path (str): The path to the CSV file containing Reddit data.
        output_dir (str): The directory to save output files.
        batch_size (int, optional): Number of texts to process at once.
                                    Adjust for optimal performance. Defaults to 1000.
    """
    # Extract the CSV filename (without extension)
    csv_name = os.path.splitext(os.path.basename(csv_file_path))[0]

    # Read the CSV file into a Pandas DataFrame
    df = read_csv_to_dataframe(csv_file_path)

    # Rename the columns
    df.columns = ['score', 'date', 'title', 'author', 'permalink', 'text', 'id', 'created_utc', 'subreddit_id']
    #print(df.head())

    # Process texts with spaCy's pipe
    texts = df["text"].tolist()
    docs = nlp.pipe(texts, batch_size=batch_size)

    # Extract entities and assign to the DataFrame
    df["entity"] = [get_most_frequent_entity(doc.text) for doc in nlp.pipe(df["text"].tolist(), batch_size=batch_size)]

    df = df.dropna(subset=['entity']).reset_index(drop=True)

    print(df.head())

    # Save the DataFrame to a CSV (without filtering)
    output_csv_path = os.path.join(output_dir, f'{csv_name}_all_entities.csv')
    df.to_csv(output_csv_path, index=False)

if __name__ == "__main__":
    # Get the input CSV file path from the command line
    if len(sys.argv) != 2:
        print("Usage: python3 your_script_name.py <input_csv_file>")
        sys.exit(1)
    csv_file_path = sys.argv[1]

    # Create an output directory
    output_dir = "OUTPUT_ALL_ENTITIES"  # Changed output directory name
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    process_reddit_data(csv_file_path, output_dir)
