# Tensorflow_Loan_Default_Prediction

DataSet obtained from Kaggle: https://www.kaggle.com/wordsforthewise/lending-club

Data can be downloaded from: https://drive.google.com/file/d/1vLF0ATXdvioYuFxAe_fIMn8kSndK5t7c/view?usp=sharing

*Given historical data on loans with label indicated that whether or not the borrower defaulted (charge-off),  we want to build a model that can predict whether or nor a borrower will pay back their loan? This way in the future when we get a new potential customer we can assess whether or not they are likely to pay back the loan.*

### For this project, we completed below tasks

**1. Data Exploratoion :** Get an understanding for which variables are important, view summary statistics, and visualize the data 

**2. Data PreProcessing :** Remove or fill any missing data. Remove unnecessary or repetitive features. Convert categorical string features to dummy variables

**3. Data Normalization:** Use a MinMaxScaler to normalize the feature data X_train and X_test

**4. Neural Network Model Build:** Use keras to build a sequential model to predict the label

**5. Model Performance Evaluation:** Plot out the validation loss versus the training loss, check precision,recall,confucion metircs


The "loan_status" column contains our label.
