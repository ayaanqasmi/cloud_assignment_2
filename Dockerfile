# Use an official Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy only requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY server.py .

# Expose port for the server
EXPOSE 8000

# Command to run the app
CMD ["python", "server.py"]
