# Company Bankruptcy Prediction

## Problem:
It is a classification problem.
As bankruptcy due to business failure can negatively affect the enterprise as 
well as the global economy, it is crucial to understand and predict whether a 
company is showing symptoms of getting bankrupt or not.
The problem statement is to develop a prediction model which will predict 
whether a company can go bankrupt or not. This will help the company to take 
appropriate decisions.

## Dataset:
The data is collected from Taiwan Economic Journal for the years 1999 to 2009. 
Company bankruptcy was defined based on the business regulations of the Taiwan 
Stock Exchange. The dataset consists of multiple financial ratio columns such 
as:

* Return on Assets (ROAs)
* Gross Profits
* Operating & Net income and Expenses
* Cash flows
* Taxes
* Growth rate
* Debt
* Turnover, Revenue, Liability, etc.

All the features are normalized in the range 0 to 1.

The target column is “Bankrupt?” (0: No, 1: Yes).

It is a highly imabalanced data.

Source : https://archive.ics.uci.edu/ml/datasets/Taiwanese+Bankruptcy+Prediction


## My Approach:
I have prepared the data in three ways:
1. Oversampled (SMOTE)
2. Undersampled (Bring down the count of majority class data)
3. Resampled (created multiple datasets containing all samples of minority class).

Then I have applied different machine learning models and Neural Networks 
from sklearn and tensorflow respectively.

There are two ipynb files in notebook directory.
Training1 : It contains my investigation on data. There are the codes and plots
            of my exploratory data analysis. The EDA is then followed by PCA 
            and SMOTE transformation. Then impelementation of ML models.
Training2 : It contains the undersampling code of data followed by different 
            models training on undersampled data. Then neural entwork build 
            and training on undersampled and oversampled data.
            Then CNN traiing on undersampled data concluded by a tabular 
            comparision.
Training3: It contains the resampling code. Here the datasets is resampled into 
            multiple balanced datasets and an ensemble model of decision trees
            is trained.
            
Saved_Models directory contains the saved weights of best model (Random Forest
in this case).

Data directory contains the data used to train the models.

