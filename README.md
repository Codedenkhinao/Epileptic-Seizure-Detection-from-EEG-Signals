# Epileptic Seizure Detection from EEG Signals

This project aims to detect epileptic seizures from EEG signals using machine learning and deep learning models such as SVM, CNN, and LSTM.

## 🚀 Project Overview

- ✅ Trained models:
  - Support Vector Machine (SVM)
  - Convolutional Neural Network (CNN)
  - Long Short-Term Memory (LSTM)
- 🧠 Dataset: (https://www.kaggle.com/datasets/harunshimanto/epileptic-seizure-recognition)
- 🗂️ Current focus: model training and evaluation

## 📊 Results (so far)

| Model | Accuracy | Notes |
|-------|----------|-------|
| SVM   | 98%      | Initial baseline |
| CNN   | 99%      | Good performance on clean data |
| LSTM  | 99%      | Promising for time-series patterns |

## Conclusion:
- SVM, CNN, and CNN-LSTM are all powerful models for the task of epileptic seizure classification. Among them, CNN and CNN-LSTM outperform SVM in terms of evaluation metrics such as accuracy, precision, recall, and F1-score.
- CNN is particularly effective for EEG data, which is inherently complex, due to its ability to automatically extract features through deep learning layers.
- While SVM is suitable for simpler problems or limited datasets, in the context of epileptic seizure detection using EEG signals, CNN and CNN-LSTM are more optimal choices in terms of performance and generalization capability.
## Limitations:
- The dataset used (Epileptic Seizure Recognition) consists of pre-processed EEG segments, each 1 second in length. Therefore, the models have not been tested on raw, continuous EEG signals as encountered in real-world scenarios.
- Although the dataset is relatively generalized, it lacks detailed information such as age, gender, and recording conditions of the patients. This may affect the model’s applicability to specific end users in practical deployment.
- Currently, the models have only been trained and evaluated in an offline environment — they have not yet been integrated or tested with any real-time hardware system.

