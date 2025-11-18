# Heart Disease Prediction Model

## Overview
This repository is my practice project for building machine learning models using health datasets from Kaggle.  
The main notebook (`final_heart_disease_model.ipynb`) is my first real end-to-end ML project, based on the famous Heart Disease dataset (originally collected by Redwan Sony et al., 2020, and combining data from Cleveland, Hungary, Switzerland, and the VA).

### Dataset
The dataset contains 14 attributes commonly used in published experiments:
- age, sex, chest pain type (cp), resting blood pressure (trestbps), serum cholesterol (chol), fasting blood sugar (fbs),  
- resting electrocardiographic results (restecg), maximum heart rate achieved (thalach), exercise-induced angina (exang),  
- oldpeak (ST depression), slope of the peak exercise ST segment, number of major vessels (ca), thalassemia (thal), and the target (presence of heart disease).

(Note: the original database had 76 attributes, but nearly all research uses this processed 14-attribute subset.)

### Notebooks
- `own_attempt.ipynb` → My very first try. I followed YouTube tutorials (mostly logistic regression) and quickly realized it wasn’t the right approach, but I kept going for the learning experience.
- `final_heart_disease_model.ipynb` → The cleaned-up, working version. I leaned on AI tools for structure but did all the debugging and improvements myself in Google Colab. Switched to Random Forest Classifier and got ~84% accuracy.

## Lessons Learned
While working on the “own attempt” notebook, I realized early that the regression model from the YouTuber wasn’t suitable. I stuck with it anyway just to finish something. Later I understood why my accuracy was terrible: I was using logistic regression on categorical features that weren’t properly encoded. Switching to Random Forest in the final version handled the categorical (“wordy”) data much better and immediately boosted performance.

## Limitations of My Model (Being Honest as a Beginner)
I got ~84% accuracy, which felt good at the time, but I now see a bunch of issues that probably make the model less trustworthy in real life:

1. **Lots of Missing Data**  
   Columns like `ca` (number of major vessels) and `thal` are missing in >50% of rows. I just filled them with mode/median, which can distort things. Next time: try iterative imputation or drop them entirely.

2. **Weird/Impossible Values**  
   Cholesterol = 0 (biologically impossible), negative `oldpeak` values, etc. I left them as-is, so the model learned from garbage in some places. Should have treated zeros as missing or clipped values.

3. **Mixed Data Sources**  
   The dataset combines Cleveland, Hungary, Switzerland, and VA Long Beach data with different missing-value patterns and zero usages. Performance almost certainly varies by location.

4. **Strong Biases**  
   ~80% male, average age ~53 → the model will probably do worse on women and younger people. Also mild class imbalance.

5. **Small Dataset + Potential Overfitting**  
   Only ~900–1000 rows after combining everything. I used a single train-test split instead of cross-validation, and some categorical variables were ordinal-encoded when they shouldn’t have been (assumes order that doesn’t exist).

These are classic beginner mistakes, but recognizing them is how I improve!  
Version 2 plans: proper cleaning, better imputation, location-based stratification, cross-validation, and trying XGBoost/LightGBM.

## Future Improvements
- Thorough data cleaning pipeline
- Separate models per clinic/location or domain adaptation
- Proper cross-validation and hyperparameter tuning
- Feature importance analysis and SHAP values
- Try neural networks just for fun

Thanks for checking out my first ML project! Feedback very welcome 🙌
