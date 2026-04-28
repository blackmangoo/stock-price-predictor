# =============================================
# Dockerfile for Stock Price Predictor
# =============================================
# This Dockerfile containerizes the Streamlit app
# so it can run on any machine without setup issues.

# Use official Python image as base
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements first (Docker caches this layer)
# This means if your code changes but dependencies don't,
# Docker won't reinstall packages - saves build time!
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project into the container
COPY . .

# Expose port 8501 (Streamlit's default port)
EXPOSE 8501

# Health check - Docker uses this to know if the app is running
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Command to run when container starts
# --server.address=0.0.0.0 makes it accessible from outside the container
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
