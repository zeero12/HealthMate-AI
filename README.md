# 🩺 HealthMate AI  
**Your AI-Powered Health Companion for Precision Care**

---

## 📌 Overview  
HealthMate AI is a smart healthcare assistant that predicts diseases based on user symptoms or skin images, and suggests relevant doctors and medications. It combines classical ML, CNN-based image classification, and LLM-powered recommendations into a single AI-driven pipeline.

---

## 🧠 Project Architecture  
### 🔁 System Flow  
The following diagrams illustrate the overall architecture and flow of the application:

![Flow 1](https://github.com/manyaj011/-HealthGuard-/assets/110671189/d38b1cab-fcc6-4368-ac89-29eac194758e)  
![Flow 2](https://github.com/manyaj011/-HealthGuard-/assets/110671189/ff35aa2e-4eea-4621-b49e-d7f70355e306)  
![Flow 3](https://github.com/manyaj011/-HealthGuard-/assets/110671189/0d21bad5-457e-47ef-9630-6f55ac8063fb)

---

## ⚙️ Technologies Used

### 💡 Machine Learning Models
- **Symptom-Based Prediction**: Decision Tree, Random Forest, XGBoost  
- **Image-Based Identification**: CNNs (Multi-class + Multiple Binary Classifiers)  
- **Chatbot**: Large Language Model (LLM)

### 📊 Algorithms Used
- Dijkstra’s Algorithm (Real-Time Mapping)
- Hash Tables with Priority Queues

### 🧰 Tech Stack
- **Languages**: Python  
- **Libraries**: TensorFlow, Keras, Scikit-learn, XGBoost, OpenCV, Plotly  
- **Other Tools**: Flask, Matplotlib, Seaborn, PIL  
- **Hardware**: NVIDIA DGX Workstation

---

## 🏃‍♂️ Running the Project

### Required Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

### Hyperparameter Tuning
```python
klearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

### CNN Architecture
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

## 🤖 Model Approaches
### 📌 If User Enters Symptoms
Models Used: Decision Tree, Random Forest, XGBoost

Tuning: Grid Search & Random Search

Best Performer: XGBoost

<img width="368" alt="XGBoost result" src="https://github.com/manyaj011/-HealthGuard-/assets/110671189/7de43367-1cea-49c7-b4e6-d0d0a37a6173">
🖼️ If User Provides Image
CNN Applied via Two Approaches:

Multi-class Classification

Multiple Binary Classifiers (one per disease)

Multi-class Results
<img width="482" alt="multi-class graph" src="https://github.com/manyaj011/-HealthGuard-/assets/110671189/21595570-2659-4951-8bc3-a6614f88826a">
Binary Classification Results (e.g., MEL)
<img width="455" alt="binary MEL graph" src="https://github.com/manyaj011/-HealthGuard-/assets/110671189/1d27eb31-65cb-403b-aa68-151ccfecdebe">
📈 Model Comparisons




💬 Doctor & Drug Recommendation via LLM
<img width="416" alt="chatbot result" src="https://github.com/manyaj011/-HealthGuard-/assets/110671189/f92a9879-283a-4304-98d5-1b1a7bc6867a">
🚀 Future Scope
🧬 1. Health Awareness Through Disease Identification
<img width="305" alt="awareness demo" src="https://github.com/manyaj011/-HealthGuard-/assets/110671189/5f06891e-15aa-4a65-9e67-d59523f33ff8">
🗺️ 2. Real-Time Disease Mapping
<img width="305" alt="realtime mapping" src="https://github.com/manyaj011/-HealthGuard-/assets/110671189/985b7f0e-8b60-4284-befc-8eb7de81c890">
📂 Datasets Used
Symptoms, Doctors, Drugs: Custom-curated dataset (included)

Images: HAM10000 Dataset

🙌 Acknowledgments
Thanks to open-source contributors and Kaggle dataset providers. Special mention to NVIDIA for access to the DGX Workstation.
