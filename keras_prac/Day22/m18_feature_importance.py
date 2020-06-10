import numpy as np

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
x = cancer.data
y = cancer.target
x_train , x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

model = DecisionTreeClassifier(n_estimators=10,max_depth=4)

model.fit(x_train,y_train)

acc = model.score(x_test,y_test)

print(acc)
print(model.feature_importances_)

def plot_feature_importance(model,target_data):
    n_features = target_data.shape[1]
    plt.barh(np.arange(n_features),model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features),cancer.feature_names)
    plt.ylabel("Features")
    plt.ylim(-1,n_features)

plot_feature_importance(model,cancer.data)
plt.show()