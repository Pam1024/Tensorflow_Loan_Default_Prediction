# Tensorflow_Loan_Default_Prediction

DataSet obtained from Kaggle: https://www.kaggle.com/wordsforthewise/lending-club

Data can be downloaded from: https://drive.google.com/file/d/1vLF0ATXdvioYuFxAe_fIMn8kSndK5t7c/view?usp=sharing

*Given historical data on loans with label indicated that whether or not the borrower defaulted (charge-off),  we want to build a model that can predict whether or nor a borrower will pay back their loan? This way in the future when we get a new potential customer we can assess whether or not they are likely to pay back the loan.*

### For this project, we completed below tasks

**1. Data Exploration :** Get an understanding for which variables are important, view summary statistics, and visualize the data 

**2. Data PreProcessing :** Remove or fill any missing data. Remove unnecessary or repetitive features. Convert categorical string features to dummy variables

**3. Data Normalization:** Use a MinMaxScaler to normalize the feature data X_train and X_test

**4. Neural Network Model Build:** Use keras to build a sequential model to predict the label

**5. Model Performance Evaluation:** Plot out the validation loss versus the training loss, check precision,recall,confucion metircs

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
- Since we will be attempting to predict loan_status, create a countplot of it
```python
sns.countplot(x='loan_status', data = df)
```

- Check the histogram of the loan_amnt column
```python
plt.figure(figsize=(10,5))
sns.distplot(df['loan_amnt'],kde = False)
```

- Explore correlation between the continuous feature variables,visualize it using a heatmap
``` python
plt.figure(figsize = (10,10))
correlation = df.corr()
sns.heatmap(correlation,cmap = "viridis",annot=True)
```

- 'loan_amnt' and 'installment' have strong correlation, check the scatterplot of them
```python
plt.figure(figsize = (10,5))
sns.scatterplot(y='loan_amnt',x='installment',data = df)
```

- Check the relationship between the loan_status and the loan_amnt using a boxplot
```python
sns.boxplot(x = 'loan_status', y = 'loan_amnt', data = df)
```

- Check the summary statistics for the loan amount based on the loan_status label

```python
df.groupby('loan_status')['loan_amnt'].describe()
```

- Check the Countplot of sub_grade  based on the loan_status label
```python
plt.figure(figsize = (12,6))
subgrade_order = sorted(df['sub_grade'].unique())
sns.countplot(x ='sub_grade',data = df, order = subgrade_order, hue = 'loan_status',palette='coolwarm')
```

**More data exploration please refer to the python file**


## 2. Data PreProcessing 


## 3. Data Normalization

## 4. Neural Network Model Build

## 5. Model Performance Evaluation
