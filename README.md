# Building the Neural Network Project - A Comprehensive Guide

Welcome to the Neural Network project! In this guide, we will walk you through the steps required to set up and run the project, providing detailed explanations and definitions to help you understand the process thoroughly.

## Initial Setup

### Cloning the Repository

To get started, you need to clone the project repository from GitHub. Open your terminal and execute the following command:

```bash
git clone https://github.com/dimapuzz1e/Neural_Network_Project.git
```

This command will create a local copy of the project on your machine. Once the cloning process is complete, navigate into the project directory using the `cd` command:

```bash
cd Neural_Network_Project
```

You are now ready to proceed with the setup.

### Data Generation

In machine learning projects, it's essential to have a dataset for both training and inference. In this project, we have a script called `data_process/data_processing.py` that takes care of generating the dataset. This script follows best practices in software design to encapsulate the dataset creation logic.

To generate the dataset, simply run the following command:

```bash
python data_process/data_processing.py
```

This script will autonomously handle the creation of the dataset required for your project.

## Training the Model

### Using Docker

For an isolated and consistent training environment, we recommend using Docker. Docker allows you to package all the necessary dependencies and configurations into a container, ensuring reproducibility.

#### Build the Training Image

To build the training environment, execute the following command:

```bash
docker build -f ./training/Dockerfile -t training_image .
```

This command will create a Docker image with all the required libraries and configurations for training.

#### Training Execution

Now that you have the training image, you can start the training process by creating a Docker container:

```bash
docker run --name (container name) training_image
```

This command will initiate the training process within an isolated environment.

#### Retrieve the Trained Model

Once the training is complete, you can retrieve the trained model from the Docker container and save it to your local directory:

```bash
docker cp (container name):/app/models/(your model name).pkl ./models/
```

Make sure to replace `your_model_name.pkl` with the actual model file name specified in your `settings.json`. To ensure everything works smoothly, create a `models` directory on your local machine:

```bash
mkdir -p ./models/
```

### Locally

If you prefer to run the training script locally, you can do so with the following command:

```bash
python3 training/training.py
```

This command will initiate the training process on your local machine.

## Inference

After training, the model is ready to make predictions on new data.

### Using Docker

For consistency, you can also set up an inference environment using Docker.

#### Set Up the Inference Image

Build the inference image by running:

```bash
docker build -t neural-inference -f inference/Dockerfile .
```

This command will create a Docker image for inference.

#### Execute Inference

To make predictions with the trained model, run the following command, which maps your local directories to the container's respective directories:

```bash
docker run -v ${PWD}/models:/app/models -v ${PWD}/data:/app/data -v ${PWD}/results:/app/results neural-inference
```

This command will execute the inference process within the Docker container, and the results will be saved directly to your local `results` directory.

### Locally

If you prefer to run inference locally, you can do so with the following command:

```bash
python inference/inference.py
```

This command will allow you to make predictions using the model you trained.

## Running Tests

To ensure the integrity and performance of your model, it's essential to run tests.

```bash
python test_neural.py
```

Before running the tests, make sure that both the data and the model are located in the expected directories as defined in your project structure.

---

Thank you for following these comprehensive instructions. If you encounter any issues, please refer to the troubleshooting guide included in the project documentation or feel free to contact the project maintainers for assistance. Happy modeling!
