# Select Python 3.9 slim image as the foundation for our container
# This image is chosen for its balance between size and utility.
FROM python:3.9-slim

# Define /app as the main directory inside the container
# All subsequent commands and operations will be based in this directory.
WORKDIR /app

# First, only copy the requirements file into the container
# This allows Docker to cache the installed packages unless the requirements file changes.
COPY requirements.txt .

# Install Python dependencies listed in the requirements file
# Using --no-cache-dir for pip keeps the Docker image size small.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the inference script, model files, data, and settings into the container
# This step transfers these essential components from your project into the Docker environment.
COPY inference/inference.py .
COPY models/ ./models
COPY data/ ./data
COPY settings.json .

# Command to execute the inference script upon container startup
# This makes the container immediately run the inference task when launched.
CMD ["python", "inference.py"]
