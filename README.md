# Propensity Score and Causal Forest Analysis

## Description

As part of the MA course *Research Design and Policy Evaluation*, the final assignment involved applying modern quantitative methods to validate or challenge the findings of a published economics paper. I selected *Natural Resources and Local Communities: Evidence from a Peruvian Gold Mine* (2013), which examines whether the expansion of one of Peru's largest gold mines increased household income in surrounding districts.

---

## Approach I — Machine Learning for Propensity Score Estimation
`ML_PropensityScore.py`

I train machine learning models to predict the probability of a district being affected to the mine expansion based on pre-treatment socioeconomic characteristics (distance to the mine, skilled workforce, access to basic services, poverty rate, public expenditure, among others). Four classifiers are evaluated — Logistic Regression with RBF kernel, Random Forest, Decision Tree, and Gradient Boosting — using nested cross-validation to avoid data leakage between hyperparameter tuning and model evaluation. The best-performing model is selected based on outer cross-validation log-loss. Propensity scores are then trimmed to the region of common support, ensuring that treated and control districts share comparable pre-treatment characteristics.

---

## Approach II — Causal Forest for Heterogeneous Treatment Effects
`GRF_HTE.R`

A more systematic approach to identifying heterogeneous treatment effects is to let the data reveal which district characteristics most strongly drive treatment effect heterogeneity, without imposing prior structure on the analysis. Thus, I estimate the Conditional Average Treatment Effect (CATE) usign Causal Forest for each district. The impact of mine expansion on household income is driven by head household educational attainment and age, and household size. 