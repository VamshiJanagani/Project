# Project
                                  Loan Status Prediction Using Exploratory Data Analysis.
                                                 
                                                
Abstract  : 

In India, the number of people applying for loans gets increased for various reasons in recent years. The bank employees are not able to analyze or predict whether the customer can pay back the amount or not (good customer or bad customer) for the given interest rate. The aim is to find the nature of the client applying for a personal loan.

The result of the analysis shows that short term loans are preferred by the majority of the clients and the clients majorly apply loans for debt consolidation. The results are shown in graphs that help the bankers to understand the client’s behavior.

Project Flow :

1.     Installing the required packages and libraries.

2.     Importing the required libraries for the model to run.

3.     Downloading the dataset, feeding it to the model, and understanding the dataset

4.     Data Preprocessing – Checking for outliers and null values. If there any null values we use Label Encoding to convert then into binary format.

5.     Dividing the model into Train and Test data. Fitting the model and predicting.

6.    Building Flask Web Application.

                                 Importing The Required Libraries  

In this milestone, we first import the packages required for building the model

1.     import pandas as pd
2.     import numpy as np
3.     from collections import Counter as c
4.     import matplotlib.pyplot as plt
5.     from sklearn import preprocessing 
6.     import seaborn as sns
7.     from sklearn.model_selection import train_test_split
8.     from sklearn.linear_model import LogisticRegression


Pandas- It is a fast, powerful, flexible, and easy to use open-source data analysis and manipulation tool, built on top of the Python programming language.
Numpy- It is an open-source numerical Python library.
Matplotlib- Visualisation with python
sklearn. Preprocessing- This package provides several common utility functions and transformer classes 
Seaborn as sns- Seaborn is a data visualization library in Python based on matplotlib  
                  
                                 Data Collection

ML depends heavily on data, without data, it is impossible for a machine to learn. It is the most crucial aspect that makes algorithm training possible. In Machine Learning projects, we need a training data set. It is the actual data set used to train the model for performing various actions.

                                                                     Download The Dataset

                                                                     Load The Dataset

1.       data=pd.read_csv("credit_train.csv")
2.       data.shape
(100514, 19)

So here we will be loading the dataset to our model. The dataset here is in CSV format. And check the number of rows and columns present in our dataset using shape attributes.

                                 Data Preprocessing

Data pre-processing is a process of cleaning the raw data i.e. the data is collected in the real world and is converted to a clean data set. In other words, whenever the data is gathered from different sources it is collected in a raw format and this data isn’t feasible for the analysis.
Therefore, certain steps are executed to convert the data into a small clean data set, this part of the process is called as data pre-processing Follow the following steps to process your Data

                                 Finding The Missing Values

Sometimes you may find some data are missing in the dataset. We need to be equipped to handle the problem when we come across them. 
One of the most common ideas to handle the problem is to take a mean of all the values of the same column and have it to replace the missing data. 

We will be using isnull().sum() method to see which column has how many null values


1.        data.isnull().sum()

Loan ID                           514
Customer ID                       514
Loan Status                       514
Current Loan Amount               514
Term                              514
Credit Score                    19668
Annual Income                   19668
Years in current job             4736
Home Ownership                    514
Purpose                           514
Monthly Debt                      514
Years of Credit History           514
Months since last delinquent    53655
Number of Open Accounts           514
Number of Credit Problems         514
Current Credit Balance            514
Maximum Open Credit               516
Bankruptcies                      718
Tax Liens                         524
dtype: int64

                                  Label Encoding And One Hot Encoding

1.        data.dropna(subset=['Loan Status'],inplace=True)
2.        le=preprocessing.LabelEncoder()
3.        data['Loan Status']=le.fit_transform(data['Loan Status'])

Here, we apply label encoding and replace the null values.

Sklearn provides a very efficient tool for encoding the levels of categorical features into numeric values. Label Encoder encodes labels with a value between 0 and n_classes-1 where n is the number of distinct labels.

If a label repeats it assigns the same value as assigned earlier. Here ‘le’ is taken as a function. It encodes the target column Loan Status, fit label encoder will return the encoded labels. 

1.        data['Term'].replace(("Short Term","Long Term"),(0,1),inplace=True)
2.        data.head()

Next, we simply replace 0 and 1 for short term and the long term where it is giv

                          Normalizing The Values

1.           data['Credit Score']=data['Credit Score'].apply(lambda val:(val /10) if val>850 else val)

Scaling is done to normalize the data within a particular range. Here we are normalizing the data for credit score by applying the lambda function.
Normalization- Normalization is the process of reorganizing data in a database so that it meets two basic requirements.
1.           do_nothing=lambda:None
2.           cscoredf=data[data['Term']==0]
3.           stermAVG=cscoredf['Credit Score'].mean()
4.           lscoredf=data[data['Term']==1]
5.           ltermAVG=lscoredf['Credit Score'].mean()
6.           data.loc[(data.Term==0) & (data['Credit Score'].isnull()),'Credit Score']=stermAVG
7.           data.loc[(data.Term==1) & (data['Credit Score'].isnull()),'Credit Score']=ltermAVG

1.           data['Credit Score'] = data['Credit Score'].apply(lambda val: "Poor" if np.isreal(val)
                                                  and val < 580 else val)
2.           data['Credit Score'] = data['Credit Score'].apply(lambda val: "Average" if np.isreal(val)
                                                  and (val >= 580 and val < 670) else val)
3.           data['Credit Score'] = data['Credit Score'].apply(lambda val: "Good" if np.isreal(val) 
                                                  and (val >= 670 and val < 740) else val)
4.           data['Credit Score'] = data['Credit Score'].apply(lambda val: "Very Good" if np.isreal(val) 
                                                  and (val >= 740 and val < 800) else val)
5.           data['Credit Score'] = data['Credit Score'].apply(lambda val: "Exceptional" if np.isreal(val) 
                                                  and (val >= 800 and val <= 850) else val)
Here, we will be handling the null values for the column credit score by using the mean method.
As we analyze we see that short term=0 and long term=1.

                                  Analyzing The Data

Firstly prints the sum of missing values in the column Annual Income.
Then, by using fillna function we are filling the null values with the mean method inplace where it is true.
Then, we are finding the data shape.
By using the counter function we are to get the count of Good, Very Good and Average.

1.           data['Credit Score'].value_counts().sort_values(ascending = True).plot(kind='bar', title ='Number of loans in terms of Credit Score category')
<AxesSubplot:title={'center':'Number of loans in terms of Credit Score category'}>

1.           print("There are",data['Annual Income'].isna().sum(), "Missing Annual Income Values.")
There are 19154 Missing Annual Income Values.
2.           # By appplying mean we fill the null values
3.           data['Annual Income'].fillna(data['Annual Income'].mean(), inplace=True)
4.           data.shape
(100000, 19)
5.           from collections import Counter as c
6.           print(c(data['Credit Score']))  #returns the class count values
Counter({'Good': 75506, 'Very Good': 18479, 'Average': 6015})
7.           data['Credit Score'] = le.fit_transform(data['Credit Score'])  #applying label encoder
c(data['Credit Score'])
8.           Counter({1: 75506, 2: 18479, 0: 6015})
9.           data['Home Ownership'].value_counts().sort_values(ascending = True).plot(kind='bar', title="Number of Loan based on Home ownership")
<AxesSubplot:title={'center':'Number of Loan based on Home ownership'}>
            
                                               Handling Missing Values
 
 Here also we are filling the null values using the fillna() method. The column here is Years In Current Job
 
                                                              Normalizing
                                                              
                                                            Outlier Detection
                                                            
                                                        Separating Independent And Dependent Variables
                                                        
                                                            Building Model
                                                            
                                                            
Once the pre-processing of data is done next we apply the train data to the algorithm.
There are several Machine learning algorithms to be used depending on the data you are going to process such as images, sound, text, and numerical values. The algorithms that you can choose according to the objective that you might have it may be Classification algorithms are Regression algorithms.

Example:     1. Linear Regression.

                    2. Logistic Regression.

                    3. Random Forest Regression / Classification.

                    4. Decision Tree Regression / Classification.

You will need to train the datasets to run smoothly and see an incremental improvement in the prediction rate.

Now we apply the Decision Tree algorithm on our dataset.

                                           Splitting The Dataset And Predicting
 
Here we split the data into x_train, x_test, y_train, and y_test. The training size data should always be three times more than the testing size data to get accurate results. So the size is less here in the code. And then we use decision tree and fit the model.
                                  
                                              Save The Model
 
We import the pickle file and dump the model into it.

Dumping the model into pickle file and saving it

                                           Build Flask Application
                                                         
                                                         
Flask Frame Work with Machine Learning Model In this section, we will be building a web application that is integrated into the model we built. A UI is provided for the uses where he has to enter the values for predictions. The enter values are given to the saved model and prediction is showcased on the UI.
