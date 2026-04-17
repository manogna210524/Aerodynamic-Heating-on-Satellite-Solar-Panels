# ☀️ Solar Panel Max Temperature Prediction Dashboard

An end-to-end Machine Learning project that predicts the **maximum temperature of solar panels** using environmental factors such as irradiance and ambient temperature.

This project combines **data preprocessing, multiple ML models, evaluation metrics, and an interactive dashboard** to provide insights into solar panel performance.

---

## 🚀 Live Dashboard

🔗 **Explore the Project Here:**
👉 https://manogna210524.github.io/Aerodynamic-Heating-on-Satellite-Solar-Panels/

> Interactive dashboard with model comparison, 3D visualization, and performance insights.

---

## 🚀 Project Overview

Efficient temperature prediction is crucial for optimizing solar panel performance and preventing overheating.

This project uses a dataset containing:

* 🌞 **Irradiance**
* 🌡️ **Ambient Temperature**
* 🔥 **Maximum Panel Temperature (Target)**

Multiple regression models are trained and compared to identify the most accurate approach.

---

## 📊 Dataset

* Format: CSV
* Features:

  * `Irradiance`
  * `Ambient Temperature`
* Target:

  * `Max Temperature (K)`

---

## ⚙️ Methodology

### 🔹 Data Preprocessing

* Cleaned and structured dataset
* Feature-target separation
* Train-test split:

  * **80% Training**
  * **20% Testing**

### 🔹 Model Evaluation Strategy

* Cross-validation applied for robustness
* Performance metrics used:

  * **R² Score**
  * **MAE (Mean Absolute Error)**
  * **RMSE (Root Mean Squared Error)**

---

## 🤖 Models Implemented

* 📈 Multiple Linear Regression
* 🌲 Decision Tree
* 🌳 Random Forest
* ⚡ AdaBoost
* 🔍 K-Nearest Neighbors (KNN)
* 🎯 Support Vector Regression (RBF Kernel)

---

## 🏆 Results & Performance

| Model                          |   R² Score |   MAE (K) |  RMSE (K) |
| ------------------------------ | ---------: | --------: | --------: |
| **Multiple Linear Regression** | **0.9641** | **3.425** | **4.303** |
| SVR (RBF)                      |     0.9635 |     3.441 |     4.336 |
| KNN (k=5)                      |     0.9577 |     3.769 |     4.670 |
| Random Forest                  |     0.9572 |     3.700 |     4.695 |
| AdaBoost                       |     0.9519 |     3.968 |     4.976 |
| Decision Tree                  |     0.9347 |     4.451 |     5.800 |

### ✅ Best Model:

**Multiple Linear Regression** performed the best with:

* Highest R² score
* Lowest MAE & RMSE

---

## 📊 Dashboard Features

* 📌 Model selection interface
* 📌 Performance comparison
* 📌 3D visualization of predictions
* 📌 Residual analysis
* 📌 Feature importance insights

---

## 📈 Key Insights

* Solar panel temperature is strongly influenced by irradiance and ambient temperature.
* Linear relationships dominate, which is why **Linear Regression outperformed complex models**.
* Ensemble models did not significantly outperform simpler approaches due to dataset simplicity.

---

## 🛠️ Tech Stack

* Python 🐍
* Scikit-learn
* Pandas & NumPy
* Plotly / Visualization Tools
* HTML/CSS (Dashboard UI)

---

## 📂 Project Structure

```
├── data/
│   └── dataset.csv
├── models/
├── notebooks/
├── dashboard/
├── README.md
```

---

## ▶️ How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the project:

   ```bash
   python app.py
   ```

---

## 📌 Future Improvements

* Add more environmental features (wind speed, humidity)
* Deploy as a scalable web app
* Integrate real-time sensor data
* Explore deep learning approaches

---

