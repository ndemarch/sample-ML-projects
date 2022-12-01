**Hotel Cancellations** *(hotel_cancellations.ipynb)*

A significant number of hotel bookings are called off due to cancellations or no-shows. Typical reasons for cancellations include 
change of plans, scheduling conflicts, etc. This is often made easier by the option to do so free of charge or preferably at a low cost. 
This may be beneficial to hotel guests, but it is a less desirable and possibly revenue-diminishing factor for hotels to deal with. Such 
losses are particularly high on last-minute cancellations.


Four machine learning models are tested (logistic regression, SVM, decision tree and random forest) to provide four useful business recommendations 
to prevent cancellations.

**Fraus Audit & Bankruptcy** *(fraud_audit_and_bankruptcy.ipynb)*

Predicting likelihood of a company going bankrupt for potential investments along with detecting the risk of fraus for clients. Three classification models are used to asses risk of fraud: Random forest with hyperparameter tuning, XGBoost with hyperparameter tuning and a Dense Artifical Neural Network. We reach a maximum accuracy of 99.1% on validation data. For predicting bankruptcy, random forest and logistic regression with thresholding are used to reach 100% accuracy on the validation data.
