# Use Python 3.8 as the base image
FROM python:3.8-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory's contents to /app in the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir flask pandas scikit-learn

# Expose port 5000 for the Flask app
EXPOSE 5000

# Run the Flask application
CMD ["python", "app.py"]
