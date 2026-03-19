# 📊 Monday BI Agent

An **AI-powered Business Intelligence (BI) assistant** that transforms monday.com data into actionable insights using a combination of **deterministic analytics + natural language querying**.

---

## 🚀 Overview

Monday BI Agent bridges the gap between **traditional dashboards** and **AI assistants**.

- Dashboards → static, require manual interpretation  
- AI tools → flexible, but often not grounded in real data  

👉 This project combines both:
- Uses **real monday.com data**
- Computes **actual business metrics**
- Adds an **AI layer for natural language interaction**

---

## ✨ Features

- 🔗 Fetches live data from **monday.com GraphQL API**
- 📊 Converts board data into **structured DataFrames**
- 📈 Computes key business metrics:
  - Total Pipeline
  - Weighted Forecast
  - Revenue
  - Work Order Status
- 💬 Natural language query interface
- 🧠 Intent classification (pipeline, revenue, operations, etc.)
- ⚙️ Rule-based fallback when AI model is unavailable
- 📉 Sector and status-based breakdowns
- 🖥️ Interactive UI using Streamlit

---

## 🧠 How It Works

The system is designed in **layers**:

### 1. Data Ingestion
- Fetches board data using monday.com API
- Parses items into structured format

### 2. Data Processing
- Converts data into Pandas DataFrames
- Handles flexible column matching (e.g., "value", "amount", "probability")

### 3. Metrics Engine (Deterministic)
- Calculates:
  - Pipeline value
  - Weighted forecast
  - Revenue
  - Operational metrics

### 4. Query Understanding (AI Layer)
- Classifies user queries into intents:
  - Pipeline
  - Revenue
  - Operations
  - Leadership summary
  - Sector analysis

### 5. Response Generation
- Maps intent → corresponding metrics
- Returns structured and human-readable output

---

## 💡 Key Design Decision

> **AI is NOT responsible for business logic.**

- All calculations → deterministic Python logic  
- AI → only interprets user queries  

✅ Ensures:
- Reliability  
- Accuracy  
- Trustworthiness  

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit** (UI)
- **Pandas** (data processing)
- **monday.com GraphQL API**
- **Hugging Face Transformers** (optional AI layer)

---

## 📦 Installation

```bash
git clone https://github.com/chhandak72/monday_bi_agent.git
cd monday_bi_agent
pip install -r requirements.txt
