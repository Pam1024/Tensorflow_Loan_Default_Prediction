# Tensorflow_Loan_Default_Prediction

DataSet obtained from Kaggle: https://www.kaggle.com/wordsforthewise/lending-club

Data can be downloaded from: https://drive.google.com/file/d/1vLF0ATXdvioYuFxAe_fIMn8kSndK5t7c/view?usp=sharing

*Given historical data on loans with label indicated that whether or not the borrower defaulted (charge-off),  we want to build a model that can predict whether or nor a borrower will pay back their loan? This way in the future when we get a new potential customer we can assess whether or not they are likely to pay back the loan.*

### For this project, we completed below tasks

#### 1. Data Exploration :
      Get an understanding for which variables are important, view summary statistics, and visualize the data 

#### 2. Data PreProcessing :
     Remove or fill any missing data. Remove unnecessary or repetitive features. Convert categorical string features to dummy variables

#### 3. Data Normalization:
     Use a MinMaxScaler to normalize the feature data X_train and X_test

#### 4. Neural Network Model Build:
     Use keras to build a sequential model to predict the label

#### 5. Model Performance Evaluation:
     Plot out the validation loss versus the training loss, check precision,recall,confucion metircs


*Here is the information on this particular data set:*
*The "loan_status" column contains our label*

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Features</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>loan_amnt</td>
      <td>The listed amount of the loan applied for by the borrower. If at some point in time, the credit department reduces the loan amount, then it will be reflected in this value.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>term</td>
      <td>The number of payments on the loan. Values are in months and can be either 36 or 60.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>int_rate</td>
      <td>Interest Rate on the loan</td>
    </tr>
    <tr>
      <th>3</th>
      <td>installment</td>
      <td>The monthly payment owed by the borrower if the loan originates.</td>
    </tr>
    <tr>
      <th>4</th>
      <td>grade</td>
      <td>LC assigned loan grade</td>
    </tr>
    <tr>
      <th>5</th>
      <td>sub_grade</td>
      <td>LC assigned loan subgrade</td>
    </tr>
    <tr>
      <th>6</th>
      <td>emp_title</td>
      <td>The job title supplied by the Borrower when applying for the loan.*</td>
    </tr>
    <tr>
      <th>7</th>
      <td>emp_length</td>
      <td>Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years.</td>
    </tr>
    <tr>
      <th>8</th>
      <td>home_ownership</td>
      <td>The home ownership status provided by the borrower during registration or obtained from the credit report. Our values are: RENT, OWN, MORTGAGE, OTHER</td>
    </tr>
    <tr>
      <th>9</th>
      <td>annual_inc</td>
      <td>The self-reported annual income provided by the borrower during registration.</td>
    </tr>
    <tr>
      <th>10</th>
      <td>verification_status</td>
      <td>Indicates if income was verified by LC, not verified, or if the income source was verified</td>
    </tr>
    <tr>
      <th>11</th>
      <td>issue_d</td>
      <td>The month which the loan was funded</td>
    </tr>
    <tr>
      <th>12</th>
      <td>loan_status</td>
      <td>Current status of the loan. This column contains our label</td>
    </tr>
    <tr>
      <th>13</th>
      <td>purpose</td>
      <td>A category provided by the borrower for the loan request.</td>
    </tr>
    <tr>
      <th>14</th>
      <td>title</td>
      <td>The loan title provided by the borrower</td>
    </tr>
    <tr>
      <th>15</th>
      <td>zip_code</td>
      <td>The first 3 numbers of the zip code provided by the borrower in the loan application.</td>
    </tr>
    <tr>
      <th>16</th>
      <td>addr_state</td>
      <td>The state provided by the borrower in the loan application</td>
    </tr>
    <tr>
      <th>17</th>
      <td>dti</td>
      <td>A ratio calculated using the borrower’s total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrower’s self-reported monthly income.</td>
    </tr>
    <tr>
      <th>18</th>
      <td>earliest_cr_line</td>
      <td>The month the borrower's earliest reported credit line was opened</td>
    </tr>
    <tr>
      <th>19</th>
      <td>open_acc</td>
      <td>The number of open credit lines in the borrower's credit file.</td>
    </tr>
    <tr>
      <th>20</th>
      <td>pub_rec</td>
      <td>Number of derogatory public records</td>
    </tr>
    <tr>
      <th>21</th>
      <td>revol_bal</td>
      <td>Total credit revolving balance</td>
    </tr>
    <tr>
      <th>22</th>
      <td>revol_util</td>
      <td>Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit.</td>
    </tr>
    <tr>
      <th>23</th>
      <td>total_acc</td>
      <td>The total number of credit lines currently in the borrower's credit file</td>
    </tr>
    <tr>
      <th>24</th>
      <td>initial_list_status</td>
      <td>The initial listing status of the loan. Possible values are – W, F</td>
    </tr>
    <tr>
      <th>25</th>
      <td>application_type</td>
      <td>Indicates whether the loan is an individual application or a joint application with two co-borrowers</td>
    </tr>
    <tr>
      <th>26</th>
      <td>mort_acc</td>
      <td>Number of mortgage accounts.</td>
    </tr>
    <tr>
      <th>27</th>
      <td>pub_rec_bankruptcies</td>
      <td>Number of public record bankruptcies</td>
    </tr>
  </tbody>
</table>

## 1. Data Exploration
- Check the general information of data
```python
df.info()
```

![info](https://github.com/Pam1024/Tensorflow_Loan_Default_Prediction/blob/main/image/info.PNG) 

- Since we will be attempting to predict loan_status, create a countplot of it
```python
sns.countplot(x='loan_status', data = df)
```
![info](https://github.com/Pam1024/Tensorflow_Loan_Default_Prediction/blob/main/image/countplot1.PNG) 

- Check the histogram of the loan_amnt column
```python
plt.figure(figsize=(10,5))
sns.distplot(df['loan_amnt'],kde = False)
```
![info](https://github.com/Pam1024/Tensorflow_Loan_Default_Prediction/blob/main/image/2histogram.PNG) 

- Explore correlation between the continuous feature variables,visualize it using a heatmap
``` python
plt.figure(figsize = (10,10))
correlation = df.corr()
sns.heatmap(correlation,cmap = "viridis",annot=True)
```
![info](https://github.com/Pam1024/Tensorflow_Loan_Default_Prediction/blob/main/image/3heatmap.PNG) 

- 'loan_amnt' and 'installment' have strong correlation, check the scatterplot of them
```python
plt.figure(figsize = (10,5))
sns.scatterplot(y='loan_amnt',x='installment',data = df)
```
![info](https://github.com/Pam1024/Tensorflow_Loan_Default_Prediction/blob/main/image/4scatterplot.PNG) 

- Check the relationship between the loan_status and the loan_amnt using a boxplot
```python
sns.boxplot(x = 'loan_status', y = 'loan_amnt', data = df)
```
![info](https://github.com/Pam1024/Tensorflow_Loan_Default_Prediction/blob/main/image/5boxplot.PNG) 

- Check the summary statistics for the loan amount based on the loan_status label

```python
df.groupby('loan_status')['loan_amnt'].describe()
```
![info](https://github.com/Pam1024/Tensorflow_Loan_Default_Prediction/blob/main/image/7summary_statistics.PNG) 

- Check the Countplot of sub_grade  based on the loan_status label
```python
plt.figure(figsize = (12,6))
subgrade_order = sorted(df['sub_grade'].unique())
sns.countplot(x ='sub_grade',data = df, order = subgrade_order, hue = 'loan_status',palette='coolwarm')
```
![info](https://github.com/Pam1024/Tensorflow_Loan_Default_Prediction/blob/main/image/8countplot.PNG) 

**More data exploration please refer to the python file**


## 2. Data PreProcessing 
- check the total count of missing values per column
```python
df.isnull().sum()
```
![info](https://github.com/Pam1024/Tensorflow_Loan_Default_Prediction/blob/main/image/9missingvalue.PNG) 

- Check how many unique employment job titles are there
```python
df['emp_title'].nunique()
```
the result is :173105, there are too many unique values, which is not very informative for the prediction, so drop this feature
```python
df = df.drop('emp_title', axis =1)
```
- Check which column is most highly correlates to mort_acc
```python
df.corr()['mort_acc'].sort_values()
```
![info](https://github.com/Pam1024/Tensorflow_Loan_Default_Prediction/blob/main/image/10mort_acc_corr.PNG) 

Looks like the total_acc feature correlates with the mort_acc ,fill in the missing mort_acc values based on their total_acc value.

If the mort_acc is missing, then we will fill in that missing value with the mean value corresponding to its total_acc value

```python
dic = df.groupby('total_acc')['mort_acc'].mean().to_dict()
def match_mort (x,y):
    if np.isnan(x):
        return dic[y]
    else:
        return x
df['mort_acc']=df.apply(lambda x:match_mort(x['mort_acc'],x['total_acc']),axis=1)
```
- Deal with  Categorical Variables 
List all the columns that are currently non-numeric
```python
df.select_dtypes('object').columns
```
![info](https://github.com/Pam1024/Tensorflow_Loan_Default_Prediction/blob/main/image/11categorical_feature.PNG) 

Deal with 'term' feature

![info](https://github.com/Pam1024/Tensorflow_Loan_Default_Prediction/blob/main/image/12term_feature.png) 

- Convert 'sub_grade' Categorical Variables to dummy variables

![info](https://github.com/Pam1024/Tensorflow_Loan_Default_Prediction/blob/main/image/13sub_grade.PNG) 
```python
subgrade_dummies = pd.get_dummies(df['sub_grade'],drop_first=True)
df = pd.concat([df.drop('sub_grade',axis=1),subgrade_dummies],axis=1)
df.columns
```
![info](https://github.com/Pam1024/Tensorflow_Loan_Default_Prediction/blob/main/image/14columns.PNG) 

- Deal with 'address' feature
Feature engineer a zip code column from the address in the data set. Create a column called 'zip_code' that extracts the zip code from the address column.
![info](https://github.com/Pam1024/Tensorflow_Loan_Default_Prediction/blob/main/image/15zip_code.PNG) 

**More data PreProcessing please refer to the python file**


## 3. Data Normalization

- Train Test Split
```python
X = df.drop('loan_repaid',axis=1).values
y = df['loan_repaid'].values
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=101)
```
- Normalizing the Data
Use a MinMaxScaler to normalize the feature data X_train and X_test
```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
```

## 4. Neural Network Model Build
**build a a model that goes 78 --> 39 --> 19--> 1 output neuron with dropout layer**
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping
model = Sequential()

#input layer
model.add(Dense(units=78,activation='relu'))
model.add(Dropout(0.2))

#hidder layer
model.add(Dense(units=39,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(units=19,activation='relu'))
model.add(Dropout(0.2))

#binary_classification 
#output layer
model.add(Dense(units=1,activation='sigmoid'))

#model compile
model.compile(optimizer='adam',loss='binary_crossentropy')

# add earlystop to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
```
fit the model
```python
model.fit(X_train,y_train,batch_size=256,epochs=50,validation_data=(X_test,y_test),callbacks=[early_stop])
```

## 5. Model Performance Evaluation
- check the how the losses of traing and testing envolve
```python
losses = pd.DataFrame(model.history.history)
losses.plot()
```
![info](https://github.com/Pam1024/Tensorflow_Loan_Default_Prediction/blob/main/image/16lossesenvolve.PNG) 

![info](https://github.com/Pam1024/Tensorflow_Loan_Default_Prediction/blob/main/image/17metrics.PNG) 

From above metrics, we can see that this model is good at predicting class'1', but not that good at predicting class'0'. The reason is that the inbalance of original dataset.
