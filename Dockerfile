# Use official Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Expose port (FastAPI default is 8000)
EXPOSE 8000

# Command to run the API
# Note: For production use gunicorn with uvicorn workers
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
