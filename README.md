# CPE312 - ML Deployment Lab

## Project Overview

This project demonstrates how to deploy a Machine Learning
model as a web application using Flask and Docker.

The system uses Random Forest Regression trained on the
California Housing dataset to predict median house values.

## Technologies Used

- Python 3.11
- Scikit-learn
- Flask
- Docker
- HTML, CSS, JavaScript

## Project Structure

- app.py - Flask API and web application
- train_model.py - Machine Learning model training
- requirements.txt - Python dependencies
- Dockerfile - Docker configuration
- templates/main.html - Web interface

## How to Run

### 1. Build Docker Image

docker build -t ds-ml-service .

### 2. Run Docker Container

docker run -d --name cpe312-ml -p 5000:5000 ds-ml-service

### 3. Open Web Application

http://localhost:5000

### 4. Test Prediction API

Endpoint: POST /predict

Example request:

{
  "features": [8.3, 25, 6, 1, 1200, 3, 34.2, -118.3]
}

Example result:

{
  "prediction": 4.038992599999997
}

The prediction is expressed in units of 100,000 USD.

## Test Results

- Docker image built successfully.
- Flask application started successfully.
- REST API prediction tested successfully.
- Prediction result: 4.038992599999997

## Screenshots

Add screenshots of the web interface, API test,
and running Docker container here.
