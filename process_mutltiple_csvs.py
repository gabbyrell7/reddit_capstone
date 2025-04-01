import os
import subprocess
import sys

# Get the directory of the current script
script_directory = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the original script
reddit_analysis_script = os.path.join(script_directory, "complaint_spacy-data_dump-single_entity.py")  

# List of CSV files to process
csv_files = [
    "FILTERED_SUBMISSIONS_TEST/RS_2018-01.csv",
    "FILTERED_SUBMISSIONS_TEST/RS_2018-02.csv",
    # Add more CSV file paths here
]

for csv_file in csv_files:
    full_csv_path = os.path.join(script_directory, csv_file)  # Ensure full path
    print(f"Processing: {full_csv_path}")
    try:
        # Use subprocess to run the original script with the CSV file as an argument
        subprocess.run([sys.executable, reddit_analysis_script, full_csv_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error processing {csv_file}: {e}")
    except FileNotFoundError:
        print(f"Error: Script not found at {reddit_analysis_script}")
        break  # Exit the loop if the script is not found
