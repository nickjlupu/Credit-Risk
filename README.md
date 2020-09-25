# Credit-Risk

## Background
Credit risk is an inherently unbalanced classification problem, as the number of good loans easily outnumber the number of risky loans. Therefore, you’ll need to employ different techniques to train and evaluate models with unbalanced classes. Jill asks you to use imbalanced-learn and scikit-learn libraries to build and evaluate models using resampling. Your final task is to evaluate the performance of these models and make a recommendation on whether they should be used to predict credit risk.

### Objectives
The goals of this challenge are to:

* Implement machine learning models.
* Use resampling to attempt to address class imbalance.
* Evaluate the performance of machine learning models.

## Resources
* Python 3.7
* Jupyter Notebook
* Libraries:  pandas, numpy, pathlib, collections, scikit-learn, imblearn
* Data: LoanStats_2019Q1.csv (data provided from LendingClub, a peer-to-peer lending services company)

## Analysis
The .csv data is read into a pandas DataFrame where it is cleaned and prepared (e.g. textual data converted to numeric).  It is then split into features (X) and target (y) variables as well as training and testing data.  We then set up four different machine learning models where the data is fit.  After each model we assess balanced accuracy score, precision, recall, and f1 score.  

## Results

### Balanced Accuracy Scores
* Naive Random Oversampling:  0.650
* SMOTE Oversampling:  0.662
* Cluster Centroids Undersampling:  0.547
* Combination (Over/Under) Sampling:  0.677

The 2 oversampling methods seem to have similar results.  The accuracy of the undersampling method is much lower than the other methods.  Finally, the combination sampling method shows a slight improvement over the other methods.    

### Precision
In all of our models the precision for high risk loans is 0.01 and 1.00 for low risk loans.  This tells us that we can confidently assess low risk loans as such.  However, many of our predictions of high risk loans may actually be low risk.  

### Recall (Sensitivity)
Recall provides the ratio of correctly predicted outcomes to the total actual outcomes in that classification.  That is, for all Low Risk Loans how many did we correctly predict as Low Risk. 

* Naive Random Oversampling:  High Risk = 0.69  Low Risk = 0.61 
* SMOTE Oversampling:  High Risk = 0.63  Low Risk = 0.69
* Cluster Centroids Undersampling:  High Risk = 0.68  Low Risk = 0.41
* Combination (Over/Under) Sampling:  High Risk = 0.78  Low Risk = 0.57


### Final Recommendation
It is important that we correctly predict high risk loans so that we can avoid defaults.  Thus, it would be best to consider using the model that maximizes recall for the high risk category (classification 0).  In this case, the Combination Sampling (SMOTEENN) model performs the best out of the four.  However, by using this model we would incur many false positives (low risk loans classified as high risk) since the precision is low.  There is an opportunity cost of lost profit by not making more of these low risk loans.  Perhaps this model could be used to rule out loans deemed high risk and run another model to further refine the low risk loans.  