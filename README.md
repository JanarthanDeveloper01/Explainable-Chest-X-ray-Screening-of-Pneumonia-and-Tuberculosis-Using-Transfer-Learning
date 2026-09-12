Explainable Chest X-ray Screening of Pneumonia and Tuberculosis Using Transfer Learning

An AI-powered chest X-ray screening system that classifies Normal, Pneumonia, and Tuberculosis using transfer learning, Grad-CAM explainability, and Llama 3.2 (Ollama) for AI-assisted clinical report generation through a Streamlit application. 
IEEE_Final_2026.pdf

Overview

This project develops an end-to-end explainable AI pipeline for chest X-ray disease screening by comparing four ImageNet-pretrained deep learning architectures under identical training conditions and selecting the best-performing model for deployment. The final system combines classification, visual explainability, confidence-based severity labeling, AI-generated summaries, and PDF report generation. 
IEEE_Final_2026.pdf

Key Features

Three-class classification: Normal, Pneumonia, Tuberculosis

Comparative evaluation of four transfer learning models

DenseNet121 selected as the proposed backbone

Grad-CAM visualization for model interpretability

AI-assisted report generation using Llama 3.2 (Ollama)

Interactive Streamlit deployment

Confidence-based severity labeling

Downloadable PDF reports

IEEE_Final_2026.pdf
 
IEEE_Final_2026.pdf

Deep Learning Models Compared

Model

	

Accuracy




ResNet50

	

87.60%




MobileNetV2

	

93.02%




EfficientNetV2B0

	

94.24%




DenseNet121 (Proposed)

	

95.23%

DenseNet121 achieved the highest performance and was selected as the deployed model. 
IEEE_Final_2026.pdf

Technologies Used

Python

TensorFlow

Keras

DenseNet121

ResNet50

MobileNetV2

EfficientNetV2B0

Grad-CAM

Streamlit

Llama 3.2 (Ollama)

Scikit-learn

IEEE_Final_2026.pdf
 
IEEE_Final_2026.pdf

Project Structure
Explainable-Chest-X-ray-Screening-of-Pneumonia-and-Tuberculosis-Using-Transfer-Learning/

├── DenseNet121.ipynb
├── Efficientnet.ipynb
├── MobileNetV2.ipynb
├── ResNet50_Model.ipynb
├── README.md
└── .gitignore
Methodology

The workflow includes:

Chest X-ray image preprocessing

Image resizing to 224×224

Backbone-specific normalization

Data augmentation

Transfer learning with four pretrained CNNs

Comparative model evaluation

Grad-CAM visualization

Confidence-based severity labeling

AI-assisted report generation using Llama 3.2

Streamlit deployment

IEEE_Final_2026.pdf
 
IEEE_Final_2026.pdf

Results

The proposed DenseNet121 model achieved:

95.23% Accuracy

0.95 Precision

0.96 Recall

0.95 F1-Score

The evaluation also demonstrated strong separation between Pneumonia and Tuberculosis predictions while concentrating most remaining errors around the Normal–Pneumonia boundary. 
IEEE_Final_2026.pdf

Explainable AI

Grad-CAM generates heatmaps highlighting the image regions that influence model predictions, improving transparency and supporting human interpretation of the results. 
IEEE_Final_2026.pdf

Streamlit Deployment

The deployed application provides:

X-ray image upload

Real-time prediction

Per-class confidence scores

Grad-CAM visualization

AI-generated clinical summary

PDF report download

IEEE_Final_2026.pdf
How to Run

Clone the repository.

Install the required Python packages.

Open the desired Jupyter Notebook.

Train or evaluate the selected model.

Launch the Streamlit application for inference.

Future Improvements

Multi-site dataset validation

Edge-device optimization

Calibrated uncertainty estimation

Clinical validation with radiologists

IEEE_Final_2026.pdf
Disclaimer

This project is intended for research and educational purposes only and is not a clinical diagnostic tool. Medical decisions should always be made by qualified healthcare professionals.

Author

Janarthan B

B.Tech Information Technology

St. Joseph's Institute of Technology, Chennai
