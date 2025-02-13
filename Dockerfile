# Use the official Python image as the base image
FROM python:3.13

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY . /app

# Install the dependencies
# RUN pip install -r requirements.txt

# Expose the port
EXPOSE 5000

# Start the API
CMD ["python", "app.py"]