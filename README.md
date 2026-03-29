# 🌍 Multi-Scale Drought Prediction in Maharashtra

A district-level drought forecasting system using **remote sensing, climate data, and deep learning** to enable accurate early warning across Maharashtra.

---

## 📌 Overview

This project builds a **multi-scale drought prediction framework** that integrates:

* Satellite data (NDVI, LST, rainfall, ET)
* Soil moisture (multi-depth)
* Temporal lag features (1–3 months)
* Machine Learning + Deep Learning models

Unlike traditional approaches, this system captures both **spatial variability (district-level)** and **temporal dynamics (time-series dependencies)** to improve prediction accuracy.

---

## 🧠 Why Multi-Scale?

The system operates across multiple dimensions:

* **Spatial:** District-level predictions across 36 regions
* **Temporal:** Monthly data (2015–2024) with lag-based forecasting
* **Data:** Multi-source integration (CHIRPS, MODIS, NASA FLDAS)
* **Model:** ML (RF, XGBoost) + DL (LSTM)

This allows modeling of **delayed environmental responses and non-linear climate interactions**, leading to significantly better performance.

---

## ⚙️ Methodology

* Data preprocessing: aggregation, interpolation, normalization
* Feature engineering: SPI/SPEI, NDVI anomalies, lag features
* Models:

  * Random Forest
  * XGBoost
  * LSTM (best performing)
* Evaluation: Accuracy, Precision, Recall, F1 Score

---

## 🏆 Results

* **LSTM achieved ~92% accuracy**
* Strong performance in capturing temporal dependencies
* Improved detection of moderate and extreme drought conditions

---

## 🗂️ Project Structure

```
Maharashtra_Drought_Prediction/
│── data/
│── notebooks/
│── src/
│── models/
│── README.md
```

---

## 📄 Research Paper

**Multi-Scale Drought Forecasting in Maharashtra using Integrated Remote Sensing, Climate Data, and Deep Learning Framework**

* Status: *Under Review (Springer format)*
* Available in /docs/

---

## 🙌 Acknowledgment

Special thanks to **Rubyscape** and their team for providing a platform that made:

* Model development
* Workflow design
* Experimentation

more efficient and streamlined.

---

## 👨‍💻 Contributors

* Yashoda Varma
* Arya Gonnade
* Sudhanshu Narvekar

---

## 🔮 Future Work

* Transformer-based models
* Higher-frequency forecasting
* Real-time dashboard deployment

---

## 📜 License

MIT License
