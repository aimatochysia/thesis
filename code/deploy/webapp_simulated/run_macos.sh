#!/bin/bash 
echo "Building Docker image..." 
docker build -t ecg-flask-app . 
echo "Running the container..." 
docker run -p 5000:5000 ecg-flask-app