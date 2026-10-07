Rice Leaf Disease Classification (PRCP-1001)
An end-to-end Computer Vision pipeline built with Python, TensorFlow, and Keras to automate the detection and classification of rice leaf diseases from digital imagery.

📌 Project Overview
Agricultural disease diagnosis relies heavily on manual visual inspection, which can be time-consuming and prone to human error. This project implements a Deep Learning framework utilizing Convolutional Neural Networks (CNN) to automatically categorize rice plant leaves into three distinct health conditions:

Bacterial Leaf Blight

Brown Spot

Leaf Smut

🛠️ Key Features
Automated Data Pipeline: Configured dynamic dataset ingestion via tf.keras.utils.image_dataset_from_directory with an 80/20 train-validation split across 119 total image samples.

Pixel Normalization & Preprocessing: Resized input images to standard 180×180 RGB dimensions and cast pixel values to float32 arrays normalized to [0,1] range for faster GPU compute convergence.

Exploratory Data Analysis & Inspection: Audited tensor batch shapes ((32,180,180,3)) and generated visual batch grids using Matplotlib and NumPy to evaluate class label distributions.

🚀 Tech Stack
Language: Python 3.x

Deep Learning Framework: TensorFlow 2.x / Keras

Data Processing & Visualization: NumPy, Matplotlib

Development Environment: Google Colab / Jupyter Notebooks

📂 Project Structure
Plaintext
├── Data/
│   ├── Bacterial leaf blight/
│   ├── Brown spot/
│   └── Leaf smut/
├── RiceLeaf.ipynb       # Main Google Colab Notebook
└── README.md            # Project Documentation
⚙️ How to Run
Clone the Repository:

Bash
git clone https://github.com/your-username/rice-leaf-disease-classification.git
cd rice-leaf-disease-classification
Mount Google Drive & Set Path:
Open RiceLeaf.ipynb in Google Colab and ensure your dataset folder is uploaded to Google Drive:

Python
from google.colab import drive
drive.mount('/content/drive')
Execute Notebook:
Run all cells sequentially to setup the environment, split data, normalize tensors, and train the model.
