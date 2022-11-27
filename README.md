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
![15](https://user-images.githubusercontent.com/93427278/204125884-c31f5fad-a769-4df8-96d7-c9adf0feeb26.png)


![16](https://user-images.githubusercontent.com/93427278/204125892-c8ad8fef-18b4-4148-b408-3efa24080534.png)


![3](https://user-images.githubusercontent.com/93427278/204125899-d0f84c26-38aa-47b7-af14-1b0c1068913c.png)


![6](https://user-images.githubusercontent.com/93427278/204125957-7bb1c2f0-039a-442a-b876-a96119e074b0.png)


![9](https://user-images.githubusercontent.com/93427278/204125941-679c2da7-f366-417c-95c0-0314493545ff.png)


![11](https://user-images.githubusercontent.com/93427278/204125966-470b4ca5-63a9-4577-a5f9-294586b1695f.png)


![13](https://user-images.githubusercontent.com/93427278/204125967-2852c73b-94ec-4bf4-82e4-5b28cce5db27.png)

## Result:

Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
