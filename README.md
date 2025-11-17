**Heart Disease Prediction Model**

Overview
This repository is for practising making machine learning models using health datasets from Kaggle. The final_heart_disease_model.ipynb is my first project, based on the dataset by Redwan Sony et al., 2020.
Dataset
The dataset consists of 14 attributes: age, sex, chest pain type, resting blood pressure, serum cholesterol, fasting blood sugar, resting electrocardiographic results, maximum heart rate achieved, exercise-induced angina, oldpeak (ST depression induced by exercise relative to rest), the slope of the peak exercise ST segment, number of major vessels, and Thalassemia.
Although the full database includes 76 attributes, all published studies (including this one) use a subset of just these 14.
Notebooks

own_attempt.ipynb: This is my initial attempt at building the model, guided by YouTube tutorials and some AI tools. I followed along but made adjustments based on what I learned.
final_heart_disease_model.ipynb: This is the refined version, where I relied more on AI for the core structure but handled debugging myself to make it work better in Google Colab.

Lessons Learned
While working on the "own attempt" notebook, I realized early on that the regression model from the YouTuber wasn't suitable for my dataset. I decided to continue with it anyway for the learning experience. Later, I figured out why my accuracy scores were so poor: I had used logistic regression, which works best for binary data, but my dataset included categorical features (like words or labels).
For the final model, switching to a random forest classifier handled the "wordy" data much better, leading to significantly higher accuracy scores.
