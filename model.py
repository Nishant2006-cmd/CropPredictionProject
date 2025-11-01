import pickle
import pandas as pd
from sklearn.model_selection import train_test_split# sk learn and train test fnx
from sklearn.ensemble import RandomForestClassifier#modl = randomforest a supervised classified model accuracy best

data = pd.read_csv("Crop_recommendation.csv")

data.head()

data.shape

data.isnull().sum()

#split feathers and labels

x =  data.iloc[:,:-1]#feacthers
y= data.iloc[:,-1]#labeel

X_train, X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state= 42)#split into traing and testing
#testing 80% ,test 20% thats why we use 0.2 its 20%
#random code 42 ensure same split

X_train.head()

y_train.head()

model =RandomForestClassifier()

model.fit(X_train,y_train)
pickle.dump(model,open('model.pkl','wb'))
predictions = model.predict(X_test)

accuracy = model.score(X_test,y_test)

print("Accuracy:",accuracy)

new_feature = [[36,58,25,28.66024,59.31891,8.399136,36.9263]]
predictied_crop = model.predict(new_feature)
print("predictied crop:",predictied_crop)