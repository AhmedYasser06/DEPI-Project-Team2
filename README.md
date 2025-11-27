# DEPI-Project-Team2  
# 🏭 AI-Powered Predictive Maintenance for Industrial Equipment

> **Predicting failures today to protect industries tomorrow**

An enterprise-grade AI-driven predictive maintenance system that identifies equipment failures before they occur using IoT sensor data, reducing downtime by up to 40% and maintenance costs by 30%.

**DEPI Project – Team 2**

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Performance](#-performance)
- [Project Structure](#-project-structure)
- [Tech Stack](#%EF%B8%8F-tech-stack)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Model Performance](#-model-performance)
- [Results & Analysis](#-results--analysis)
- [Deployment](#-deployment)
- [Contributing](#-contributing)
- [Team](#-team)
- [License](#-license)

---

## 🎯 Overview

This project implements a comprehensive predictive maintenance solution that leverages machine learning to forecast equipment failures using real-time IoT sensor data. By analyzing patterns in vibration, temperature, pressure, humidity, and operational load, the system provides early warnings of potential failures.

### Business Impact

- ✅ **Reduce Downtime**: Prevent unexpected equipment failures
- ✅ **Lower Costs**: Decrease maintenance expenses by 30%
- ✅ **Optimize Operations**: Schedule maintenance efficiently
- ✅ **Extend Lifespan**: Increase equipment lifetime by 25%
- ✅ **Improve Safety**: Enhance workplace safety standards

### Key Capabilities

- **Real-time Monitoring**: Process IoT sensor streams continuously
- **Advanced ML Models**: Ensemble of 7 optimized algorithms
- **High Accuracy**: 96.18% overall accuracy with 70.87% F1-score on failures
- **Interpretable Results**: SHAP-based explanations for predictions
- **Production Ready**: Complete deployment pipeline with API

---

## ✨ Key Features

### Machine Learning Pipeline

- **Multiple Model Training**: XGBoost, CatBoost, LightGBM, Random Forest
- **Ensemble Methods**: Stacking and Voting classifiers
- **Hyperparameter Optimization**: Automated tuning with Optuna (40 trials)
- **Class Imbalance Handling**: SMOTE, ADASYN, BorderlineSMOTE
- **Threshold Optimization**: Custom threshold selection per model
- **Probability Calibration**: Isotonic and Platt scaling

### Feature Engineering

- **70+ Engineered Features**: Advanced interaction terms
- **Time-based Features**: Machine age, operational hours, maintenance recency
- **Sensor Fusion**: Multi-sensor stress indices and patterns
- **Risk Scoring**: Composite risk assessment algorithms
- **Domain-specific Ratios**: Oil-to-coolant, power consumption rates

## 🏆 Performance

### Best Model: CatBoost Optimized

| Metric        | Score      | Description                              |
| ------------- | ---------- | ---------------------------------------- |
| **F1-Score**  | **0.7087** | Primary optimization target              |
| **Precision** | **0.6475** | 64.75% of predicted failures are correct |
| **Recall**    | **0.7826** | Catches 78.26% of actual failures        |
| **ROC-AUC**   | **0.9822** | Excellent discrimination capability      |
| **Accuracy**  | **96.18%** | Overall prediction accuracy              |

### Confusion Matrix Breakdown

```
                          Predicted

                      No Failure  Failure

Actual      No        68,650      1,897
            Failure   968         3,485
```

- **True Positives**: 3,485 (correctly predicted failures)
- **True Negatives**: 68,650 (correctly predicted healthy)
- **False Positives**: 1,897 (false alarms - 2.69%)
- **False Negatives**: 968 (missed failures - 21.74%)

---

## 📂 Project Structure

```
AI-Predictive-Maintenance/
│
├── data/                          # Data directory
│   ├── raw/                       # Original datasets
│   ├── Splited Data/              # Train, Val, Test
│
│
├── notebooks/                     # Jupyter notebooks
│   ├── EDA.ipynb              # Exploratory data analysis
├── src/                           # Source code
│   ├── data_preparing.py         # Data loading & feature engineering
│   ├── data_training.py          # Model training & optimization
│   ├── data_evaluation.py        # Model evaluation & visualization
│   ├── loading_model.py          # Model loading & inference
│   ├── loading_test.py           # Test suite
│   └── main_pipeline.py          # Main orchestration script
│
├── models/                        # Saved models
│   ├── best_model_catboost_optimized.pkl
│   ├── model_threshold.json
│   ├── scaler_optimized.pkl
│   ├── feature_info.json
│   └── README.md
│
├── reports/
│   ├── milestone 1
│   ├── milestone 2
│   ├── Final Report
│
│
├── deployment/                    # Deployment files
│   ├── api/                      # API implementation
│   │   ├── app.py               # FastAPI application
│   │   └── requirements.txt     # API dependencies
│   ├── docker/                   # Docker configuration
│   │   ├── Dockerfile
│   │   └── docker-compose.yml
│   └── inference/                # Inference scripts
│       └── predict.py
└── README.md
```

---

## 🛠️ Tech Stack

### Core Technologies

| Category                | Technologies                              |
| ----------------------- | ----------------------------------------- |
| **Language**            | Python 3.8+                               |
| **ML Frameworks**       | Scikit-learn, XGBoost, CatBoost, LightGBM |
| **Data Processing**     | NumPy, Pandas                             |
| **Visualization**       | Matplotlib, Seaborn, Plotly               |
| **Imbalanced Learning** | Imbalanced-learn (SMOTE, ADASYN)          |
| **Optimization**        | Optuna                                    |
| **Interpretability**    | SHAP                                      |

### Deployment Stack

| Component               | Technology             |
| ----------------------- | ---------------------- |
| **API Framework**       | FastAPI / Flask        |
| **Containerization**    | Docker, Docker Compose |
| **Experiment Tracking** | MLflow                 |
| **Model Format**        | Pickle, ONNX           |
| **Monitoring**          | Prometheus, Grafana    |
| **CI/CD**               | GitHub Actions         |

## 📊 Model Performance

### Model Comparison

| Rank | Model                  | Threshold | F1-Score   | Precision | Recall | ROC-AUC |
| ---- | ---------------------- | --------- | ---------- | --------- | ------ | ------- |
| 🥇   | **CatBoost Optimized** | 0.50      | **0.7087** | 0.6475    | 0.7826 | 0.9822  |
| 🥈   | CatBoost Calibrated    | 0.30      | 0.7029     | 0.6154    | 0.8194 | 0.9818  |
| 🥉   | Voting Ensemble        | 0.63      | 0.7022     | 0.6232    | 0.8042 | 0.9818  |
| 4    | LightGBM Optimized     | 0.89      | 0.7017     | 0.6214    | 0.8060 | 0.9818  |
| 5    | Stacking Ensemble      | 0.86      | 0.6935     | 0.6167    | 0.7923 | 0.9593  |
| 6    | XGBoost Optimized      | 0.56      | 0.6896     | 0.6317    | 0.7593 | 0.9807  |
| 7    | Random Forest          | 0.81      | 0.6670     | 0.5626    | 0.8190 | 0.9770  |

### Classification Report (Best Model)

```
                precision    recall  f1-score   support

   No Failure      0.9861    0.9731    0.9796     70547
      Failure      0.6475    0.7826    0.7087      4453

     accuracy                           0.9618     75000
    macro avg      0.8168    0.8778    0.8441     75000
 weighted avg      0.9653    0.9618    0.9631     75000
```

### ROC Curve Performance

All models demonstrate excellent discrimination with ROC-AUC > 0.95:

- **CatBoost Optimized**: 0.9822
- **Voting Ensemble**: 0.9818
- **LightGBM Optimized**: 0.9818

---

## 🔍 Results & Analysis

### Error Analysis

#### False Positives (1,897 samples)

- **Impact**: Unnecessary maintenance alerts
- **Average Probability**: 0.7010
- **Probability Range**: 0.50 → 0.96
- **Cost**: Lower priority, causes inefficiency

#### False Negatives (968 samples) ⚠️

- **Impact**: Critical - Missed actual failures
- **Average Probability**: 0.2958
- **Probability Range**: 0.00 → 0.50
- **Analysis**: Samples exhibit normal-like behavior, challenging to classify

#### True Positives (3,485 samples)

- **Success Rate**: 78.26% recall
- **Average Probability**: 0.7814
- **Confidence**: High confidence predictions

### Feature Importance (Top 10)

Based on SHAP analysis:

1. **Operational_Hours** (18.2%)
2. **Temperature_C** (15.7%)
3. **Vibration_mms** (12.4%)
4. **Stress_Index** (9.8%)
5. **Maintenance_Rate** (8.3%)
6. **Power_per_Hour** (7.1%)
7. **Oil_Level_pct** (6.5%)
8. **Error_Density** (5.9%)
9. **Failure_History_Count** (5.2%)
10. **Last_Maintenance_Days_Ago** (4.8%)

### Threshold Analysis

| Threshold | Precision  | Recall     | F1-Score   | Use Case              |
| --------- | ---------- | ---------- | ---------- | --------------------- |
| 0.30      | 0.5521     | 0.8854     | 0.6806     | Maximize recall       |
| **0.50**  | **0.6475** | **0.7826** | **0.7087** | **Balanced**          |
| 0.70      | 0.7329     | 0.6195     | 0.6714     | Minimize false alarms |
| 0.90      | 0.8542     | 0.3126     | 0.4580     | High precision        |

---

## 🚢 Deployment

### REST API Deployment

Start the prediction API:

```bash
cd deployment/api
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

#### API Endpoints

**Health Check**

```bash
GET http://localhost:8000/health
```

**Single Prediction**

```bash
POST http://localhost:8000/predict
Content-Type: application/json

{
  "Temperature_C": 85.5,
  "Vibration_mms": 12.3,
  "Sound_dB": 82.1,
  ...
}
```

**Batch Prediction**

```bash
POST http://localhost:8000/predict/batch
Content-Type: multipart/form-data

file: sensor_data.csv
```

### Docker Deployment

```bash
# Build image
docker build -t predictive-maintenance:latest -f deployment/docker/Dockerfile .

# Run container
docker run -p 8000:8000 predictive-maintenance:latest

# Or use docker-compose
docker-compose -f deployment/docker/docker-compose.yml up
```

### Docker Compose

```yaml
version: "3.8"
services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
    environment:
      - MODEL_PATH=/app/models
```

---

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_models.py -v

# Run with coverage
pytest --cov=src tests/
```

### Test Model Loading

```bash
python src/loading_test.py --model_dir ./models/
```

---

## 📈 Future Enhancements

### Planned Features

- [ ] **Real-time IoT Integration**: MQTT/OPC-UA protocol support
- [ ] **SHAP Dashboard**: Interactive model interpretability
- [ ] **Edge Deployment**: ONNX optimization for IoT devices
- [ ] **Multi-modal Failures**: Extend to additional failure types
- [ ] **Deep Learning**: RNN/Transformer time-series models
- [ ] **Anomaly Detection**: Unsupervised learning integration
- [ ] **A/B Testing**: Model versioning and comparison
- [ ] **Alert System**: Email/SMS notifications for predictions
- [ ] **Mobile App**: iOS/Android monitoring application

### Research Directions

- Transfer learning from similar industrial domains
- Federated learning for privacy-preserving training
- Causal inference for maintenance recommendations
- Multi-task learning for failure type classification

---

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make your changes**
4. **Add tests** for new functionality
5. **Commit your changes**
   ```bash
   git commit -m "Add amazing feature"
   ```
6. **Push to the branch**
   ```bash
   git push origin feature/amazing-feature
   ```
7. **Open a Pull Request**

### Code Standards

- Follow PEP 8 style guide
- Add docstrings to all functions
- Include type hints
- Write unit tests (>80% coverage)
- Update documentation

### Reporting Issues

Please use GitHub Issues for bug reports and feature requests. Include:

- Clear description
- Steps to reproduce
- Expected vs actual behavior
- Environment details

---

## 👥 Team

**DEPI Project – Team 2**

| Role                | Responsibility                    |
| ------------------- | --------------------------------- |
| **Data Scientists** | Model development & optimization  |
| **ML Engineers**    | Pipeline engineering & deployment |

