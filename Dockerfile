# Use the official Python image as the base image
FROM python:3.6.1

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements2.txt .

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements2.txt

# Copy the project files into the container
COPY . .

# Set the entry point for the container
