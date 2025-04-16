# AI-Driven Facial Recognition System using Mathematical Feature Extraction

This is a Flask-based facial recognition system that integrates **AI-driven models** with **mathematical feature extraction techniques** like **Principal Component Analysis (PCA)** and **Linear Discriminant Analysis (LDA)** to identify faces with high accuracy and performance.

> **Project Type:** Research Thesis  
> **Tech Stack:** Python · Flask · PCA · LDA · CNN · SVM · MongoDB

---

## 📌 Features

- 📷 Image Capture & Upload
- 🔎 Facial Preprocessing
- 🧠 PCA & LDA Feature Extraction
- 🧠 CNN Model for Deep Feature Learning
- ✅ SVM Classifier for Identity Prediction
- 🔐 Admin Dashboard for User Logs & Monitoring
- 🗂️ MongoDB/NoSQL Database Integration
- 📊 Confidence Score & Identity Verification

---

## 🖼️ System Architecture

- plaintext
User Upload Image ─▶ Preprocessing ─▶ PCA/LDA ─▶ CNN Model ─▶ SVM Classifier ─▶ Result

---

## 🔧 System Requirements

### ✅ Hardware
- Intel i5/i7 or AMD Ryzen
- RAM: 8GB+ (16GB Recommended)
- GPU: NVIDIA GTX 1050 or better (optional for training CNN)

### ✅ Software
- OS: Windows 10/11 or Ubuntu Linux
- Python 3.8+
- MongoDB or PostgreSQL

---

## ⚙️ Installation Guide

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/ai-face-recognition-math-extraction.git
cd ai-face-recognition-math-extraction

```Create env
python -m venv venv
source venv/bin/activate        # for Linux/Mac
venv\Scripts\activate           # for Windows

``` Install dependencies required
pip install -r requirements.txt

``` Setup Database
sudo service mongod start   # Start on Windows (used for this project)
``
----

``` Run the app
python app.py

Visit: http://127.0.0.1:5000

