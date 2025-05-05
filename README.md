# ML Models to predict Parking Violations

# Introduction

The NYC Parking Violations 2015 dataset contains detailed information on parking tickets issued in New York City during the 2017 fiscal year. It includes key fields such as violation code, issue date and time, vehicle type and make, license plate state, precinct, and issuing agency. The dataset provides a real-world foundation for analyzing parking trends, identifying common violations, and understanding city enforcement patterns.

# Objectives

Predict Payment Behavior for Parking Violations: Use machine learning models in PySpark (Gradient Boosted Trees, Logistic Regression) to predict whether a parking violation will be paid or remain unpaid. Hyperparameter Tuning: Leverage PySpark’s MLib to fine-tune model parameters using TrainValidationSplit and ParamGridBuilder. Model Evaluation: Apply TrainValidationSplit and CrossValidator to rigorously test models and ensure robust generalization to unseen data.

# Platform Specifications

* Hadoop Version: 3.3.3
* Spark Version: 3.2.1
* Cluster Configuration: 5 nodes, 8 CPU cores, 860.4 GB total memory

# Dataset Specifications

* Dataset name: NYC Parking Tickets
* Dataset source URL: https://www.kaggle.com/datasets/new-york-city/nyc-parking-tickets?select=Parking_Violations_Issued_-_Fiscal_Year_2017.csv 
* Country: USA
* Number of Files: 1
* Dataset size: 2.1 GB
