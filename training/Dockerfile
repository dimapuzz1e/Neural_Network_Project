# Start with a lightweight Python 3.9 base image, optimized for size and efficiency
FROM python:3.9-slim

# Set the working directory inside the container to /app. 
# This is where all commands will be run inside the container.
WORKDIR /app

# Copy the training scripts, data, and settings into the container.
# This step transfers the necessary files from the host machine to the container's filesystem.
COPY training/ ./training
COPY data/ ./data
COPY settings.json .

# Copy the Python dependencies file into the container.
# This file lists all external Python packages required by the application.
COPY requirements.txt .

# Install the Python dependencies from the requirements file.
# The `--no-cache-dir` flag prevents caching the downloaded packages to reduce image size.
RUN pip install --no-cache-dir -r requirements.txt

# Define the command to run the training script when the container starts.
# This is the default action when the container is run, executing the training process.
CMD ["python", "./training/training.py"]
