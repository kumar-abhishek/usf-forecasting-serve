# FROM python:3.9
# WORKDIR /app
# COPY api/requirements.txt requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt
# COPY api app
# CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.9

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY api/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the API code
COPY api app

# Copy the trained model into the container
COPY training/lgb_model.pkl app/training/lgb_model.pkl

# Expose the port
EXPOSE 8000

# Start the API
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]