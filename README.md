# ⚡ Energy Consumption Prediction Pipeline

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.x-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)

An end-to-end Machine Learning ecosystem for forecasting energy consumption. This project transforms raw time-series data into actionable insights through a robust LSTM-based pipeline, served via a high-performance FastAPI backend.

---

## 📖 Table of Contents
1. [📑 Project Overview](#-project-overview)
2. [📊 Dataset Description](#-dataset-description)
3. [🏗️ Technical Architecture](#-technical-architecture)
4. [📂 Repository Structure](#-repository-structure)
5. [🛠️ Installation & Setup](#%EF%B8%8F-installation--setup)
6. [📈 ML Workflow](#-ml-workflow)
7. [🌐 API Documentation](#-api-documentation)
8. [🐳 Containerization](#-containerization)
9. [✅ Quality Assurance](#-quality-assurance)

---

## 🚀 Project Overview

### Problem Statement
In modern grid management, accurate forecasting of energy consumption is critical for balancing supply, reducing operational costs, and minimizing environmental impact. This project implements a **Recurrent Neural Network (LSTM)** to predict consumption trends based on historical sequences, enabling proactive energy distribution.

### Core Objectives
*   Build a **scalable and modular** ML pipeline.
*   Compare **Baseline (Random Forest)** vs. **Advanced (LSTM)** models.
*   Deploy as a **RESTful API** for real-time inference.
*   Ensure **reproducibility** through environment standardization (Docker).

---

## 📊 Dataset Description

The model uses historical energy event data.
- **Source:** [Public Energy Events CSV](https://raw.githubusercontent.com/MohamadNach/Machine-Learning-to-Predict-Energy-Consumption/master/events.csv)
- **Target Variable:** `Consumption` (measured in MWh).
- **Temporal Resolution:** Hourly/Daily timestamps.
- **Exogenous Features:** Extracted during preprocessing:
    - `Year`, `Month`, `Date`
    - `Time` (Hour of day)
    - `Week` (ISO week number)
    - `Day` (Day of the week)

---

## 🏗️ Technical Architecture

### ML Pipeline Structure
1.  **Ingestion:** Automated fetching from remote CSV sources.
2.  **Transformation:**
    *   Datetime parsing and indexing.
    *   Feature Engineering (Time-based components).
    *   Standardization using `MinMaxScaler` (Range: 0-1).
3.  **Windowing:** Sequence generation with a **100-step look-back window**.
4.  **Modeling:**
    *   **Baseline:** Random Forest Regressor (Scikit-Learn).
    *   **Advanced:** Multi-layer Stacked LSTM (TensorFlow/Keras).

---

## 📂 Repository Structure

```text
├── api/                # FastAPI application & endpoints
├── data/               # Local data storage (ignored by git)
├── models/             # Serialized .h5 models & .joblib scalers
├── notebooks/          # EDA & original research artifacts
├── src/                # Modular Core Logic
│   ├── data_processing.py # Cleaning & Feature Engineering
│   ├── model.py           # Architecture definitions
│   ├── train.py           # Training orchestration
│   └── evaluate.py        # Metrics & Visualizations
├── tests/              # Pytest suite
├── Dockerfile          # Production container config
└── requirements.txt    # Dependency manifest
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Virtual Environment (recommended)

### Steps
```bash
# 1. Clone & Enter
git clone https://github.com/1Yash8/MachineLearning-Project.git
cd MachineLearning-Project

# 2. Environment Setup
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Dependency Installation
pip install -r requirements.txt
```

---

## 📈 ML Workflow

### 1. Training
The `train.py` script supports multiple model types:
```bash
# Train the LSTM model (Default)
python src/train.py --model lstm

# Train the Baseline model
python src/train.py --model baseline
```

### 2. Evaluation
Performance is measured via MAE, RMSE, and R² Score:
```bash
python src/evaluate.py
```
*Visualizations (Actual vs. Predicted) are saved to `notebooks/evaluation_plot.png`.*

---

## 🌐 API Documentation

The project includes a FastAPI server for serving model predictions.

### Start the Server
```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

### Inference Endpoint: `/predict`
**POST** `http://localhost:8000/predict`

**Request Header:** `Content-Type: application/json`

**Example Request Body:**
```json
{
  "data": [
    12300.5, 12450.2, 12000.1, ... 
  ] // Needs 100 recent time-series values
}
```

**Example Successful Response:**
```json
{
  "prediction": 12645.82,
  "unit": "MWh"
}
```

---

## 🐳 Containerization

Deploy the entire stack using Docker to ensure environment parity.

```bash
# Build image
docker build -t energy-predictor:latest .

# Run container
docker run -p 8000:8000 energy-predictor:latest
```

---

## ✅ Quality Assurance

Unit tests are implemented for the data processing module to ensure reliability.

```bash
# Run all tests
pytest tests/
```

---

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.
