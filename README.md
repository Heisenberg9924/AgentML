# AgentML: Autonomous Multi-Agent ML Scientist

AgentML is a multi-agent AI system that automates the end-to-end machine learning lifecycle. Built using **LangGraph**, the framework orchestrates specialized agents that collaboratively analyze datasets, engineer features, benchmark models, generate explanations, and provide improvement recommendations.

The goal of AgentML is to reduce manual effort in machine learning workflows while maintaining transparency, extensibility, and explainability.

---

## 🚀 Features

### 🤖 Multi-Agent Architecture

AgentML is composed of specialized agents, each responsible for a distinct stage of the ML workflow:

| Agent                     | Responsibility                                    |
| ------------------------- | ------------------------------------------------- |
| Dataset Analysis Agent    | Understands dataset structure and prediction task |
| Feature Engineering Agent | Cleans, transforms, and prepares data             |
| Experiment Agent          | Trains and benchmarks multiple ML models          |
| Critic Agent              | Reviews results and suggests improvements         |

---

### 📊 Automated Dataset Analysis

The system automatically:

* Detects classification or regression tasks
* Identifies feature types
* Analyzes missing values
* Generates dataset summaries
* Validates target columns

---

### ⚙️ Intelligent Feature Engineering

AgentML performs:

* Missing value handling
* Categorical encoding
* Numerical scaling
* Data cleaning
* Train-test splitting

without requiring manual preprocessing pipelines.

---

### 🏆 Automated Model Benchmarking

The framework benchmarks **8+ machine learning algorithms** and automatically selects the best-performing model.

#### Classification Models

* Logistic Regression
* Random Forest
* XGBoost
* CatBoost
* LightGBM
* Support Vector Machine
* K-Nearest Neighbors
* Decision Tree

#### Regression Models

* Linear Regression
* Random Forest Regressor
* XGBoost Regressor
* CatBoost Regressor
* LightGBM Regressor
* Support Vector Regressor
* KNN Regressor
* Decision Tree Regressor

---

### 📈 Automated Model Selection

AgentML automatically selects the optimal model using task-specific metrics.

#### Classification

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

#### Regression

* RMSE
* MAE
* R² Score

---

### 🔍 Explainable AI

The Explainability tool provides:

* SHAP Feature Importance
* Global Model Explanations
* Local Prediction Explanations
* Feature Ranking Reports
* Statistical Feature Analysis

This enables transparent and interpretable machine learning pipelines.

---

### 🧠 Critic & Reflection System

The Critic Agent acts as an autonomous reviewer and:

* Analyzes model performance
* Detects potential overfitting
* Reviews feature quality
* Identifies weaknesses in the pipeline
* Suggests improvement strategies

Inspired by reflection-based agent architectures, this enables iterative improvement of ML workflows.

---

### 🔌 MCP Integration

AgentML integrates the **Model Context Protocol (MCP)** to support:

* External tool access
* Dataset discovery
* Knowledge retrieval
* Autonomous research workflows
* Future web-based data acquisition

---

## 🏗️ Architecture

```text
User Dataset
      │
      ▼
Dataset Analysis Agent
      │
      ▼
Feature Engineering Agent
      │
      ▼
Experiment Agent
      │
      ▼
Explainability Agent
      │
      ▼
Critic Agent
      │
      ▼
Final Report
```

---

## 🔄 Workflow

### 1. Dataset Analysis

The Dataset Analysis Agent:

* Loads the dataset
* Identifies the target variable
* Detects problem type
* Generates dataset statistics

---

### 2. Feature Engineering

The Feature Engineering Agent:

* Handles missing values
* Encodes categorical variables
* Scales numerical features
* Produces training-ready datasets

---

### 3. Experimentation

The Experiment Agent:

* Trains multiple ML models
* Evaluates performance
* Benchmarks algorithms
* Selects the best-performing model

---

### 4. Explainability

The Explainability Tool:

* Generates SHAP explanations
* Produces feature importance reports
* Explains model decisions

---

### 5. Critique

The Critic Agent:

* Reviews experimental outcomes
* Detects weaknesses
* Generates actionable recommendations

---

## 🛠️ Tech Stack

### Core Frameworks

* Python
* LangGraph
* LangChain

### Machine Learning

* Scikit-Learn
* XGBoost
* CatBoost
* LightGBM

### Explainability

* SHAP

### Data Processing

* Pandas
* NumPy

### API & Deployment

* FastAPI

### Tool Integration

* Model Context Protocol (MCP)

---

## 📂 Project Structure

```text
AgentML/
│
├── agents/
│   ├── dataset_analysis_agent.py
│   ├── feature_engineering_agent.py
│   ├── experiment_agent.py
│   ├── base_agent.py
│   └── critic_agent.py
│
├── tools/
│   ├── dataset_tools.py
│   ├── preprocessing_tools.py
│   ├── training_tools.py
│   ├── evaluation_tools.py
│   └── explainability_tools.py
│
├── graph/
│   └── workflow.py
│
├── data/
│
├── main.py
│
└── README.md
```

---

## 📌 Example Use Cases

### Healthcare

* Disease prediction
* Patient risk assessment
* Clinical outcome forecasting

### Finance

* Credit scoring
* Fraud detection
* Risk analysis

### Retail

* Customer churn prediction
* Demand forecasting
* Recommendation systems

### Education

* Student performance prediction
* Learning analytics

---

## 📈 Current Capabilities

* ✅ Multi-agent ML workflow orchestration
* ✅ Automated feature engineering
* ✅ Benchmarking of 8+ ML algorithms
* ✅ Dynamic best-model selection
* ✅ SHAP-based explainability
* ✅ Critic and reflection system
* ✅ MCP integration support

---

## 🔮 Future Roadmap

### Autonomous Data Collection

* Web scraping through MCP
* Public dataset discovery
* Automated data ingestion

### Advanced Feature Engineering

* Automated feature generation
* Feature selection
* Statistical transformations

### Hyperparameter Optimization

* Optuna integration
* Bayesian optimization
* Experiment tracking

### Enhanced Agent Collaboration

* Agent debate systems
* Reflection loops
* Self-improving workflows

---

## ⭐ Why AgentML?

AgentML combines **AutoML**, **Explainable AI**, and **Agentic AI** into a single framework. Instead of simply training models, it acts as an autonomous ML scientist capable of understanding data, conducting experiments, explaining results, and recommending improvements.

This project demonstrates practical applications of:

* Multi-Agent Systems
* Generative AI
* Explainable AI (XAI)
* Automated Machine Learning (AutoML)
* Model Context Protocol (MCP)
* Workflow Orchestration with LangGraph
