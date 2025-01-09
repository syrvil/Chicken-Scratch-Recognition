# Build stage
FROM python:3.10-slim
WORKDIR /app

# Copy requirements.txt into the build stage
COPY requirements.txt /app/

# Upgrade pip and install dependencies
# Install torch, torchvision without CUDA to get image smaller
# Finally transformers
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir torch~=2.5.1 torchvision~=0.20.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir transformers~=4.47.1 && \
    rm -rf /root/.cache/pip

# Final stage
#FROM python:3.10-slim

# Set the working directory
#WORKDIR /app

# Copy the installed dependencies from the build stage
#COPY --from=builder /app /app

# Copy application files into the container
COPY main.py /app/
COPY model.py /app/
COPY schemas.py /app/
COPY models/ /app/models/

# Expose port 8000 for the FastAPI app
EXPOSE 8080

# Set the command to run the FastAPI app with Uvicorn
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

# To pass environment variables to the FastAPI app, use as a shell command.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port=${PORT:-8080}"]
