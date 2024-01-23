import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json
import os
import logging

# Configure logging
# Setting up logging for efficient monitoring and debugging of the process.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_iris_data(settings):
    """
    Generate and split Iris dataset into training and inference sets,
    and save them as CSV files.

    Parameters:
    - settings: Configuration for controlling data storage paths, dataset sizes, etc.
    """
    try:
        # Extracting settings from the configuration file.
        data_dir = settings['general']['data_dir']
        train_config = settings['train']
        inference_config = settings['inference']

        # Check for the existence of the data directory, create it if it doesn't exist.
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # Loading Iris data from Wikipedia.
        url = 'https://en.wikipedia.org/wiki/Iris_flower_data_set'
        tables = pd.read_html(url)
        df = tables[0]  # Assuming the required table is the first one.

        # Remove the 'Dataset order' column
        if 'Dataset order' in df.columns:
            df.drop(columns=['Dataset order'], inplace=True)

        # Assuming the target column is named 'Species' and it's categorical
        if 'Species' in df.columns:
            label_encoder = LabelEncoder()
            df['Species'] = label_encoder.fit_transform(df['Species'])

        # Splitting the dataset into training and test sets.
        # It's important to use random_state for reproducibility of the results.
        train, inference = train_test_split(df, test_size=train_config['test_size'], random_state=settings['general']['random_state'])

        # Saving the datasets in CSV format.
        train.to_csv(f'{data_dir}/{train_config["table_name"]}', index=False)
        inference.to_csv(f'{data_dir}/{inference_config["inp_table_name"]}', index=False)

        logging.info("Iris data successfully generated and saved.")

    except Exception as e:
        # Logging any exceptions that occur.
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    try:
        # Loading settings from a JSON file.
        with open('settings.json') as f:
            settings = json.load(f)
        generate_iris_data(settings)
    except FileNotFoundError:
        # Handling the error of the missing configuration file.
        logging.error("The settings.json file was not found.")
    except json.JSONDecodeError:
        # Handling JSON decoding errors.
        logging.error("Error decoding settings.json.")
    except Exception as e:
        # Logging unexpected errors.
        logging.error(f"An unexpected error occurred: {e}")
