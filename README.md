# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.
```
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Vishranthi A
RegisterNumber: 212221230124
*/
```
```
import pandas as pd
import numpy as np
df=pd.read_csv('student_scores.csv')
print(df)

X=df.iloc[:,:-1].values
Y=df.iloc[:,1].values
print(X,Y)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)

Y_pred=reg.predict(X_test)
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error , mean_squared_error

plt.scatter(X_train,Y_train,color='green')
plt.plot(X_train,reg.predict(X_train),color='purple')
plt.title(' Training set (Hours Vs Scores)')
plt.xlabel('Hours')
plt.ylabel('Scores')

plt.scatter(X_test,Y_test,color='green')
plt.plot(X_test,reg.predict(X_test),color='purple')
plt.title(' Training set (Hours Vs Scores)')
plt.xlabel('Hours')
plt.ylabel('Scores')

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print('RMSE = ',rmse)
```

## Output:
![simple linear regression model for predicting the marks scored](sam.png)
![193417358-a73ad98c-ed06-4fe7-b82e-b59476894ff8](https://user-images.githubusercontent.com/93427278/193596880-5e866dce-ee5c-4494-93a6-c95d575e7335.png)
![193417370-46219869-66de-41d7-9ebe-01759e249799](https://user-images.githubusercontent.com/93427278/193596914-eef1d455-61e5-4b5c-bb11-0b155d9896bf.png)
![193417133-86cce099-c86c-4dea-8782-99f556a686e1](https://user-images.githubusercontent.com/93427278/193596972-81302d60-6a8e-45e4-964c-1168e5ae31d3.png)
![193417404-13f82485-76af-4760-b5b1-1f313257a85e](https://user-images.githubusercontent.com/93427278/193597010-51fbbe45-bf25-4f2d-aec6-90c431effc7a.png)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
