# Project Task: ASL Keypoint-Based Sign Recognition

## Context

You are working on **isolated sign recognition**. Each training sample is a short video of a single sign, but instead of raw pixels, you’re given **MediaPipe keypoints**. 

Your task is to build a classifier that maps these **sequences of keypoints over time** into the correct labels.

## Assignment

### 1\. Data Understanding

* Download this dataset: [Sign Language Recognition (Pose Landmarks)](https://www.kaggle.com/competitions/asl-signs/data)

* Explore the dataset format (e.g., numpy arrays / parquet).

* Inspect what features are available (hand, pose, face).

* Describe any patterns or challenges you see (missing landmarks, variable sequence lengths, noise).

* (optional) Work with a subsample of the dataset to keep experiments manageable; you may choose how to subsample.

### 2\. Preprocessing / Feature Engineering

* Normalize keypoints (e.g. scale by torso length, center around nose, etc.).

* Handle sequence length variability (padding, truncation, resampling).

### 3\. Baseline and Challenger Modeling 

* Train at least one baseline classifier using a simple convolutional neural network (CNN) over the time dimension of the keypoint sequences.

* Create one or two challenger models with a method of your choice. Possible ideas include:

  * Data augmentation  
  * Improved preprocessing (e.g., more advanced normalization, feature selection)  
  * Alternative temporal models (e.g., LSTM, Transformer encoder)  
  * Any other approach you prefer

### 4\. Evaluation

* Choose one or more metrics for evaluation.

* Show a confusion matrix and highlight signs that are most often confused.

### 5\. Reflection

* Compare baseline vs challenger model performance.

* If the challenger improves performance, explain why.

## Deliverables

* **Notebook(s) / scripts** with clean, runnable code.

* **README or Google Doc** with:  
  * How to run code  
  * Summary of approach & results

## Constraints

* Keep models lightweight (should run on a single GPU or even CPU if subset).

* No need for leaderboard-level optimization — clarity and reasoning matter more.

* Unlike the original Kaggle competition, you do not need to implement deployment steps (e.g., TensorFlow Lite conversion, mobile inference, or on-device optimization). This assignment focuses only on data understanding, preprocessing, modeling, and evaluation.

