from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
x = cancer.data
y = cancer.target
x_train , x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

model = DecisionTreeClassifier(max_depth=4)

feat  = model.fit(x_train,y_train)

acc = model.score(x_test,y_test)

print(acc)
print(feat.feature_importances_)