import os
import json
import logging
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class IrisClassifier(nn.Module):
    def __init__(self):
        """ Initialize layers for the IrisClassifier neural network with improved architecture. """
        super(IrisClassifier, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.bn1 = nn.BatchNorm1d(128)  
        self.dropout = nn.Dropout(0.5)  
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 3)

    def forward(self, x):
        """ Define the forward pass of the neural network. """
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

def prepare_data(train_path, test_size, random_state):
    """ Prepare data for training and testing. """
    data = pd.read_csv(train_path)
    X = data.drop('Species', axis=1).values
    y = LabelEncoder().fit_transform(data['Species'])
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def create_data_loaders(X_train, y_train):
    """ Create DataLoader for training. """
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    return DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)

def evaluate_model(model, X_test, y_test):
    """ Evaluate the model's performance on the test set. """
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
        accuracy = accuracy_score(y_test.cpu(), predicted.cpu())
    return accuracy

def train_model(settings):
    """ Train the IrisClassifier model with given settings. """
    try:
        data_dir, train_config, models_dir, model_path = extract_settings(settings)

        # Corrected the scope issue by defining train_path before its use
        train_path = f"{data_dir}/{train_config['table_name']}"

        X_train, X_test, y_train, y_test = prepare_data(train_path, train_config['test_size'], settings['general']['random_state'])
        train_loader = create_data_loaders(X_train, y_train)

        model = IrisClassifier()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        perform_training(train_loader, model, criterion, optimizer, X_test, y_test, epochs=50)
        save_model(model, models_dir, model_path)

    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
    except Exception as e:
        logging.error(f"An error occurred during training: {e}")


def perform_training(train_loader, model, criterion, optimizer, X_test, y_test, epochs):
    """ Perform the training process. """
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            train_batch(model, X_batch, y_batch, optimizer, criterion)

        evaluate_epoch(model, epoch, X_test, y_test)
    end_time = time.time()
    logging.info(f"Training completed in {end_time - start_time} seconds")

def train_batch(model, X_batch, y_batch, optimizer, criterion):
    """ Train model on a single batch. """
    optimizer.zero_grad()
    outputs = model(X_batch)
    loss = criterion(outputs, y_batch)
    loss.backward()
    optimizer.step()

def evaluate_epoch(model, epoch, X_test, y_test):
    """ Evaluate model performance at the end of an epoch. """
    accuracy = evaluate_model(model, torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    logging.info(f"Epoch {epoch+1}, Accuracy: {accuracy}")

def save_model(model, models_dir, model_path):
    """ Save the trained model. """
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    torch.save(model.state_dict(), model_path)

def extract_settings(settings):
    """ Extract settings from the configuration file. """
    data_dir = settings['general']['data_dir']
    train_path = f"{data_dir}/{settings['train']['table_name']}"
    models_dir = settings['general']['models_dir']
    model_path = f"{models_dir}/{settings['inference']['model_name']}"
    return data_dir, settings['train'], models_dir, model_path

if __name__ == "__main__":
    try:
        with open('settings.json') as f:
            settings = json.load(f)
        train_model(settings)
    except FileNotFoundError:
        logging.error("The settings.json file was not found.")
    except json.JSONDecodeError:
        logging.error("Error decoding settings.json.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
