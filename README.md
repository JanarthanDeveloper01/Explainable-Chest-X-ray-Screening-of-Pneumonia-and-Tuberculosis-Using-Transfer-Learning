# 🩺 Explainable Chest X-ray Screening of Pneumonia and Tuberculosis Using Transfer Learning

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-Neural%20Networks-red?logo=keras)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20Application-FF4B4B?logo=streamlit)
![AI/ML](https://img.shields.io/badge/AI%2FML-Computer%20Vision-purple)
![Explainable AI](https://img.shields.io/badge/Explainable%20AI-Grad--CAM-green)
![LLM](https://img.shields.io/badge/LLM-Llama%203.2-blueviolet)

> **An explainable AI-based chest X-ray screening system for Normal, Pneumonia, and Tuberculosis using transfer learning, Grad-CAM, and Llama 3.2 (Ollama), deployed through Streamlit.**

---

## 📌 Overview

This project presents an end-to-end **Explainable AI (XAI)** pipeline for chest X-ray screening of:

- 🟢 **Normal**
- 🟠 **Pneumonia**
- 🔴 **Tuberculosis**

Four ImageNet-pretrained deep learning architectures were comparatively evaluated under a common training and evaluation protocol:

- ResNet50
- MobileNetV2
- EfficientNetV2B0
- DenseNet121

Based on the experimental results, **DenseNet121 was selected as the proposed and deployed backbone**, achieving **95.23% accuracy**.

The final system integrates:

**X-ray Classification → Grad-CAM Explainability → Confidence-based Severity → Llama 3.2 AI-assisted Report → PDF Patient Report**

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🩻 **Three-Class Classification** | Normal, Pneumonia, and Tuberculosis |
| 🧠 **Transfer Learning** | Four ImageNet-pretrained CNN architectures |
| 🏆 **Best Model** | DenseNet121 |
| 🔍 **Explainable AI** | Grad-CAM visual heatmaps |
| 📊 **Confidence Analysis** | Per-class prediction probabilities |
| ⚕️ **Severity Labeling** | Confidence-based severity estimation |
| 🤖 **LLM Integration** | Llama 3.2 using Ollama |
| 🌐 **Web Application** | Streamlit |
| 📄 **PDF Report** | Downloadable `Patient_Report_1.pdf` |

---

# 🧠 Deep Learning Models Compared

| Model | Accuracy | Status |
|---|---:|---|
| ResNet50 | 87.60% | Baseline |
| MobileNetV2 | 93.02% | Compared |
| EfficientNetV2B0 | 94.24% | Compared |
| 🏆 **DenseNet121** | **95.23%** | **Proposed Model** |

### 🏆 Best Model: DenseNet121

DenseNet121 achieved the highest performance across the evaluated metrics and was selected as the proposed backbone for the deployed screening system.

---

# 📊 Performance

### DenseNet121 Results

| Metric | Score |
|---|---:|
| 🟢 **Accuracy** | **95.23%** |
| 🔵 **Precision** | **0.95** |
| 🟣 **Recall** | **0.96** |
| 🟠 **F1-Score** | **0.95** |

The evaluation showed strong overall classification performance, with most remaining errors concentrated around the **Normal–Pneumonia** boundary.

---

# 🛠️ Technologies Used

### Programming & ML

![Python](https://img.shields.io/badge/Python-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-red?logo=keras)
![Scikit Learn](https://img.shields.io/badge/Scikit--learn-F7931E?logo=scikit-learn)

### Deep Learning Models

![DenseNet121](https://img.shields.io/badge/DenseNet121-95.23%25-green)
![ResNet50](https://img.shields.io/badge/ResNet50-87.60%25-blue)
![MobileNetV2](https://img.shields.io/badge/MobileNetV2-93.02%25-orange)
![EfficientNetV2B0](https://img.shields.io/badge/EfficientNetV2B0-94.24%25-purple)

### Explainability & AI

![GradCAM](https://img.shields.io/badge/Explainable%20AI-Grad--CAM-brightgreen)
![Llama](https://img.shields.io/badge/Llama%203.2-Ollama-blueviolet)

### Deployment

![Streamlit](https://img.shields.io/badge/Streamlit-Application-FF4B4B?logo=streamlit)

---

# 🔬 Methodology

The complete workflow consists of the following stages:

### 1. Dataset Construction

Two publicly available chest-radiograph collections were combined to create a three-class dataset containing:

- Normal
- Pneumonia
- Tuberculosis

Class balancing was handled during optimization using inverse-frequency class weighting.

### 2. Image Preprocessing

- Image resizing to **224 × 224**
- Backbone-specific normalization
- Data augmentation
- Rotation
- Translation
- Zoom
- Shear
- Brightness variation

### 3. Transfer Learning

Four ImageNet-pretrained architectures were evaluated:

```text
ResNet50
MobileNetV2
EfficientNetV2B0
DenseNet121
All models were trained using a common experimental protocol.

4. Model Evaluation

Models were evaluated using:

Accuracy
Precision
Recall
F1-Score
5. Model Selection

DenseNet121 achieved the best overall performance:

Accuracy  : 95.23%
Precision : 0.95
Recall    : 0.96
F1-Score  : 0.95
6. Explainable AI

Grad-CAM was integrated to generate visual heatmaps showing the regions contributing to the model's prediction.

7. AI-Assisted Reporting

The prediction output is passed to a locally hosted:

Llama 3.2 → Ollama

The model generates a constrained AI-assisted summary based on the prediction, confidence score, and severity label.

8. Streamlit Deployment

The complete workflow is integrated into a Streamlit application for interactive inference.

🔍 Explainable AI with Grad-CAM

Grad-CAM is used to visualize the image regions that contribute to the predicted class.

The system generates:

Chest X-ray
      ↓
DenseNet121 Prediction
      ↓
Predicted Class + Confidence
      ↓
Grad-CAM Heatmap
      ↓
Explainable Prediction

This improves model transparency and makes the prediction easier to interpret.

🤖 AI-Assisted Clinical Reporting

The application integrates Llama 3.2 through Ollama for AI-assisted report generation.

X-ray Image
     ↓
Model Prediction
     ↓
Confidence Score
     ↓
Severity Label
     ↓
Grad-CAM Visualization
     ↓
Llama 3.2
     ↓
AI-Assisted Summary
     ↓
PDF Patient Report

The generated report includes the prediction, confidence information, explainability output, and AI-assisted summary.

🌐 Streamlit Application

The deployed application provides:

🩻 X-ray image upload
⚡ Real-time prediction
📊 Per-class confidence scores
🔍 Grad-CAM visualization
⚕️ Confidence-based severity labeling
🤖 Llama 3.2 AI-assisted report generation
📄 Downloadable patient report
📥 PDF report generation
Generated Report

The application generates a downloadable PDF report:

Patient_Report_1.pdf

Note: Patient_Report_1.pdf is an output generated by the application. Actual patient medical images or personally identifiable medical information should not be uploaded to this public repository.

📁 Project Structure
Explainable-Chest-X-ray-Screening-of-Pneumonia-and-Tuberculosis-Using-Transfer-Learning/
│
├── DenseNet121.ipynb
├── Efficientnet.ipynb
├── MobileNetV2.ipynb
├── ResNet50_Model.ipynb
├── streamlit_app.py
├── README.md
└── .gitignore
▶️ How to Run
1. Clone the Repository
git clone https://github.com/JanarthanDeveloper01/Explainable-Chest-X-ray-Screening-of-Pneumonia-and-Tuberculosis-Using-Transfer-Learning.git
2. Navigate to the Project
cd Explainable-Chest-X-ray-Screening-of-Pneumonia-and-Tuberculosis-Using-Transfer-Learning
3. Install Dependencies
pip install tensorflow keras opencv-python scikit-learn streamlit matplotlib pandas numpy

For Llama 3.2 integration, install and configure Ollama separately.

4. Run the Streamlit Application
streamlit run streamlit_app.py
5. Run the Jupyter Notebooks

Open any of the following notebooks:

DenseNet121.ipynb
Efficientnet.ipynb
MobileNetV2.ipynb
ResNet50_Model.ipynb
📈 Research Contribution

This project combines several components into a single end-to-end screening workflow:

Deep Learning
      +
Transfer Learning
      +
Comparative Model Evaluation
      +
Explainable AI
      +
LLM Integration
      +
Web Deployment
      +
Automated PDF Reporting

The study evaluates four architectures under matched conditions and selects DenseNet121 as the proposed backbone based on experimental performance.

🚀 Future Improvements
🌍 Multi-site external dataset validation
⚡ Edge-device model optimization
📊 Calibrated uncertainty estimation
👨‍⚕️ Formal evaluation by practicing radiologists
🔬 Further optimization for real-world deployment
⚠️ Disclaimer

This project is developed for research and educational purposes only.

It is not a clinical diagnostic tool and should not be used as a substitute for professional medical diagnosis.

Medical decisions should always be made by qualified healthcare professionals.

👨‍💻 Author
Janarthan B

B.Tech Information Technology
St. Joseph's Institute of Technology, Chennai, India

Areas of Interest

AI/ML • Computer Vision • Deep Learning • Generative AI • Python • Data Science • Research & Development
