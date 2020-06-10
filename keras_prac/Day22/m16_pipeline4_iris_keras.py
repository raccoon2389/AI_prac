from sklearn.datasets import load_iris
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Conv2D, Flatten,Dense, MaxPooling2D
import numpy as np
from sklearn.pipeline import Pipeline
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,KFold,train_test_split
from sklearn.preprocessing import MinMaxScaler



#1 데이터
dataset= load_iris()
x = dataset.data
y = dataset.target
x_train,x_test , y_train,y_test = train_test_split(x,y,test_size=0.2)
y = np_utils.to_categorical(y)

def build_model(drop=0.5,optimizer = 'adam'):
    
    inputs = Input(shape=(4,), name='input')
    x = Dense(100,activation='relu',name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(50, activation='relu',name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(30, activation='relu',name='hidden3')(x)
    x = Dropout(drop)(x)
    output1 = Dense(3,activation='softmax',name='output')(x)
    model = Model(inputs=inputs,outputs=output1)
    model.compile(optimizer=optimizer,metrics=['acc'],loss="categorical_crossentropy")

    return model

def create_hyper():
    batches = [10,40,50]
    optimizer = ['rmsprop','adam','adadelta']
    # droptout = [0.1,0.2,0.3]
    droptout = np.linspace(0.1,0.5,5).tolist()
    return{"model__batch_size" : batches, "model__optimizer"  : optimizer, "model__drop": droptout}


kf = KFold()
model = KerasClassifier(build_fn=build_model, verbose = 1)

hyperparameters = create_hyper()

pipe = Pipeline([("scal",MinMaxScaler()),("model",model)])

search = RandomizedSearchCV(estimator=pipe, param_distributions=hyperparameters,n_iter=10,cv=3,n_jobs=1)

search.fit(x_train,y_train)
score = search.score(x_test,y_test)
print(search.best_params_)
print(score)