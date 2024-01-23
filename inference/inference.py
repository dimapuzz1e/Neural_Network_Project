import os
import time
import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Enhanced logging configuration for better traceability
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

class IrisClassifier(nn.Module):
    def __init__(self):
        """ 
        Constructor for the IrisClassifier.
        This neural network is architected for classifying the Iris dataset. It consists of 4 fully connected layers with batch normalization and dropout for regularization.
        """
        super(IrisClassifier, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.bn1 = nn.BatchNorm1d(128)  # Batch normalization for stabilizing learning
        self.dropout = nn.Dropout(0.5)  # Dropout to reduce overfitting
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 3)  # Output layer for 3 classes

    def forward(self, x):
        """ 
        Forward pass definition for the network.
        Applies ReLU activations after each linear transformation, except the final layer.
        """
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)  # No activation on output layer

def load_model(model_path):
    """Load a trained IrisClassifier model from the specified file path."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = IrisClassifier()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode
    return model

def softmax(x):
    """Compute the softmax of vector x in a numerically stable way."""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

def run_inference(model, data):
    """Run model inference on the provided data and return the predictions."""
    with torch.no_grad():  # Disable gradient computation for inference
        predictions = model(data)
    return predictions

if __name__ == "__main__":
    try:
        # Load configuration settings
        if not os.path.exists('settings.json'):
            raise FileNotFoundError("Settings file not found")

        with open('settings.json') as f:
            settings = json.load(f)

        # Construct paths for model and data
        model_path = os.path.join(settings['general']['models_dir'], settings['inference']['model_name'])
        inference_data_path = os.path.join(settings['general']['data_dir'], settings['inference']['inp_table_name'])

        # Validate existence of inference data
        if not os.path.exists(inference_data_path):
            raise FileNotFoundError("Inference data file not found")

        # Model loading and data preparation
        model = load_model(model_path)
        inference_data = pd.read_csv(inference_data_path)
        if inference_data.shape[1] < 4:
            raise ValueError("Inference data must have at least 4 columns")

        X_inference = torch.tensor(inference_data.iloc[:, :4].values, dtype=torch.float32)

        # Perform inference and measure time
        start_time = time.time()
        logits = run_inference(model, X_inference).numpy()
        probabilities = softmax(logits)
        predicted_classes = np.argmax(probabilities, axis=1)

        # Save results to CSV
        results = pd.DataFrame({
            'Predicted Class': predicted_classes,
            'Probability Class 0': probabilities[:, 0],
            'Probability Class 1': probabilities[:, 1],
            'Probability Class 2': probabilities[:, 2]
        })

        results_dir = settings['general']['results_dir']
        os.makedirs(results_dir, exist_ok=True)

        results_path = os.path.join(results_dir, 'inference_results.csv')
        results.to_csv(results_path, index=False)
        end_time = time.time()

        logging.info(f"Inference completed in {end_time - start_time:.2f} seconds")

    except FileNotFoundError as e:
        logging.error(e)
    except ValueError as e:
        logging.error(e)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
