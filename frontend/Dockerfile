# Stage 1: Builder
FROM python:3.10-slim AS builder

# Set the working directory in the builder stage
WORKDIR /app

# Copy requirements.txt and install dependencies in the builder stage
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --prefix=/install -r /app/requirements.txt

# Stage 2: Final Image
FROM python:3.10-slim

# Set the working directory in the final stage
WORKDIR /app

# Copy dependencies from the builder stage to the final image
COPY --from=builder /install /usr/local

# Copy the application code to the final image
COPY . /app/

# Expose the conatiner port that Streamlit will run on
# EXPOSE 8501 serves only as a documentation for the user, 
# it does not actually publish the port
# Explicit port binding is done by using docker run -p or docker-compose
EXPOSE 8501

# Set the command to run the Streamlit app
# NOTE! The server.addreas 0.0.0.0 is the docker container's private network address, 
# not the host machine's address, tough it is accessible from the host machine 0.0.0.0/127.0.0.1/localhost
# addresses as well. Container's address 127.0.0.0 or localhost, will not be accessible 
# from the host machine, because the traffic stays inside the container reragdless of port binding.
# To pass environment variables to the Streamlit app, we can use as a shell command.
CMD ["sh", "-c", "streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT:-8051}"]



