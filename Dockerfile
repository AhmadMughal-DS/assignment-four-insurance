# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose ports for Streamlit and MLflow
EXPOSE 8501
EXPOSE 5000

# Set environment variables
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV MLFLOW_TRACKING_URI=http://localhost:5000

# Run Streamlit
CMD ["streamlit", "run", "streamlit_basic.py"] 