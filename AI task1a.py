import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_excel("C:/Users/ASUS/Downloads/student_scores - student_scores.xlsx")
print("Dataset imported successfully")
dataset
dataset.plot(x='Hours',y='Scores',style='o')
plt.title("Hours vs Percentage")
plt.xlabel("Hours studied")
plt.ylabel("Percentage Score")
plt.show()
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
print("Training complete.")
print(regressor.intercept_)
print(regressor.coef_)
line=regressor.coef_*x+regressor.intercept_
plt.scatter(x,y)
plt.plot(x,line)
plt.show()
print(x_test)
y_pred=regressor.predict(x_test)
df=pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
df
from sklearn import metrics
print("Mean Absolute Error=",metrics.mean_absolute_error(y_test,y_pred))
print("Mean Squared Error=",metrics.mean_squared_error(y_test,y_pred))
print("Root Mean Squared Error=",metrics.mean_squared_error(y_test,y_pred))
hours=9.25
test=np.array([hours])
test=test.reshape(-1,1)
own_pred=regressor.predict(test)
print("No of Hours={}".format(hours))
print("Predicted Score={}".format(own_pred[0]))
