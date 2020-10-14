# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model,preprocessing
import pandas as pd
import numpy as np 


# %%
data = pd.read_csv("car.data",sep=",")
data.head()


# %%
data.columns


# %%
# Here labelencoder will take lables in file and turn them in integer values
le = preprocessing.LabelEncoder() #creating object

#This will take entire buying column, convert it in list and transform it in integers
#following will return to us a numpy arraru
buying = le.fit_transform(list(data["buying"])) 
maint = le.fit_transform(list(data["maint"])) 
door = le.fit_transform(list(data["door"])) 
persons = le.fit_transform(list(data["persons"])) 
lug_boot = le.fit_transform(list(data["lug_boot"])) 
safety = le.fit_transform(list(data["safety"])) 
cls = le.fit_transform(list(data["class"])) 

print(buying)

predict = "class"

X = list(zip(buying,maint,door,persons,lug_boot,safety))
Y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split( X, Y, test_size=0.1)
#print(x_train,y_test)

model = KNeighborsClassifier(n_neighbors=9)

model.fit(x_train,y_train)
acc = model.score(x_test,y_test)
print(acc)

predicted = model.predict(x_test)
names = ['unacc','acc','good','vgood']

for x in range(len(predicted)):
    print("predicted: ",names[predicted[x]], "data : ", x_test[x], "Acutal : ", names[y_test[x]])
    n = model.kneighbors([x_test[x]],9,True)
    print("N: ",n)


# %%


