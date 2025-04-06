import pandas as pd
import os
import sys
from collections import Counter

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
        # Optimization: Use 'usecols' to read only the 'entity' column
        df = pd.read_csv(filepath, usecols=['entity'], **kwargs)
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


def get_entity_counts(df, entity_column='entity'):
    """
    Counts the occurrences of each entity in a DataFrame column.

    Args:
        df (pd.DataFrame): DataFrame.
        entity_column (str, optional): Name of the entity column. Defaults to 'entity'.

    Returns:
        collections.Counter: A Counter object containing entity counts.
    """
    if entity_column not in df.columns:
        print(f"Error: The DataFrame must contain a '{entity_column}' column.")
        return Counter()  # Return an empty Counter to avoid errors
    # Optimization: Convert the Series to a list for slightly faster counting
    return Counter(df[entity_column].tolist())


def save_entity_counts_to_txt(entity_counts, csv_filename, output_dir, sort_by_count=True):
    """
    Saves entity counts to a text file, optionally sorting by count.

    Args:
        entity_counts (collections.Counter): Counter object with entity counts.
        csv_filename (str): Name of the original CSV file.
        output_dir (str): Directory to save the output text file.
        sort_by_count (bool, optional): Whether to sort entities by count (descending).
                                        Defaults to True.
    """
    output_txt_path = os.path.join(output_dir, f'{os.path.splitext(csv_filename)[0]}_entity_counts.txt')
    try:
        with open(output_txt_path, 'w') as f:
            f.write(f"Entity Counts from: {csv_filename}\n")
            if sort_by_count:
                sorted_entities = entity_counts.most_common()  # Sort by count (descending)
            else:
                sorted_entities = entity_counts.items()  # Keep original order

            lines = [f"{entity}: {count}\n" for entity, count in sorted_entities]
            f.writelines(lines)
        print(f"Entity counts saved to: {output_txt_path}")
    except Exception as e:
        print(f"Error saving entity counts: {e}")

def main(csv_file_path, output_dir="OUTPUT_ENTITY_COUNTS", entity_column='entity'):
    """
    Reads a CSV, analyzes entity counts, and saves the results to a text file.

    Args:
        csv_file_path (str): Path to the input CSV file.
        output_dir (str, optional): Directory to save the output.
                                   Defaults to "OUTPUT_ENTITY_COUNTS".
        entity_column (str, optional): Name of the entity column.
                                   Defaults to 'entity'.
    """

    # Optimization: Read only the 'entity' column
    df = read_csv_to_dataframe(csv_file_path)
    if df.empty:
        return

    # Analyze entity counts
    entity_counts = get_entity_counts(df, entity_column)
    if not entity_counts:
        return

    # Extract the original CSV filename
    csv_filename = os.path.basename(csv_file_path)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the results to a text file
    save_entity_counts_to_txt(entity_counts, csv_filename, output_dir)


if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python3 your_script_name.py <input_csv_file> [entity_column]")
        sys.exit(1)

    csv_file_path = sys.argv[1]
    entity_column = 'entity'

    if len(sys.argv) == 3:
        entity_column = sys.argv[2]

    main(csv_file_path, entity_column=entity_column)
