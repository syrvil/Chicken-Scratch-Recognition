#!/bin/bash

# Function to start FastAPI backend
start_backend() {
    PYTHONPATH=$(pwd)/backend uvicorn backend.main:app --reload &
    echo "FastAPI backend started."
}

# Function to start Streamlit frontend
start_frontend() {
    PYTHONPATH=$(pwd) streamlit run frontend/app.py &
    echo "Streamlit frontend started."
}

# Function to stop FastAPI backend
stop_backend() {
    pkill -f 'uvicorn backend.main:app'
    echo "FastAPI backend stopped."
}

# Function to stop Streamlit frontend
stop_frontend() {
    pkill -f 'streamlit run frontend/app.py'
    echo "Streamlit frontend stopped."
}

# Main logic to handle start and stop
case $1 in
    start)
        # Start both applications
        start_backend
        start_frontend
        ;;
    stop)
        # Stop both applications
        stop_backend
        stop_frontend
        ;;
    restart)
        # Restart both applications
        stop_backend
        stop_frontend
        start_backend
        start_frontend
        ;;
    *)
        echo "Usage: $0 {start|stop|restart}"
        exit 1
        ;;
esac
