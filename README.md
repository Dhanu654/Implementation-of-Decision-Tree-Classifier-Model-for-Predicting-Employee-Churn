# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load Data – Import the employee dataset with relevant features and churn labels.

2. Preprocess Data – Handle missing values, encode categorical features, and split into train/test sets.

3. Initialize Model – Create a DecisionTreeClassifier with desired parameters.

4. Train Model – Fit the model on the training data.

5. Evaluate Model – Predict on test data and check accuracy, precision, recall, etc.

6. Visualize & Interpret – Visualize the tree and identify key features influencing churn.
## Program and Output:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Dhanusya K
RegisterNumber:  212223230043
*/
```
~~~
import pandas as pd
data = pd.read_csv("C:\\Users\\admin\\OneDrive\\Desktop\\Folders\\ML\\DATASET-20250226\\Employee.csv")
data.head()
~~~
![436586430-4a93d1d1-a9b8-458d-8da9-9ddd0d58cbd7](https://github.com/user-attachments/assets/000ea6d4-1565-429c-930a-46a07b07d973)
~~~
data.info()
data.isnull().sum()
data['left'].value_counts()
~~~
![436586655-82dffe3d-2531-419d-a6c5-bce25e628736](https://github.com/user-attachments/assets/c2e61628-d676-4f2e-ae2c-6d84a1649751)
~~~
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['salary'] = le.fit_transform(data['salary'])
data.head()
~~~
![436587123-62a67c73-7633-4981-bdbb-796fdb7337e8](https://github.com/user-attachments/assets/73cadc61-d913-4794-9250-2e473fac7796)
~~~
x=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]
x.head()
~~~
![436587400-9aabb266-32fa-4927-ac9b-87b89d1d4e4a](https://github.com/user-attachments/assets/01345a9c-b22d-4163-a44b-cb6e273cc943)
~~~
y=data['left']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state =100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)
y_predict=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_predict)
accuracy
~~~
![436587681-092f0e0d-2f1c-41ea-8ffc-f64037a9dd23](https://github.com/user-attachments/assets/df602824-d905-4a06-86cd-4387474ba2c6)
~~~
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
~~~
![436587836-2d9f6941-339b-45f3-b186-7519446bea8b](https://github.com/user-attachments/assets/bc8439d0-5959-4d52-a5cc-3512b9ff22ec)
~~~
import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
plot_tree(dt,feature_names=x.columns,class_names=['salary','left'],filled=True)
plt.show()
~~~
![436588135-35289f4c-7227-48fa-a9aa-59c41b83202f](https://github.com/user-attachments/assets/69ab36f0-289a-4765-8a63-20f0c165c070)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
