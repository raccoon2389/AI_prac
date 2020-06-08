from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, LSTM,Dropout, Dense
import numpy as np
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.metrics import accuracy_score




#1 데이터
print(np.linspace(0.1,0.5,5))
(x_train, y_train), (x_test,y_test) = mnist.load_data()


# x_train = x_train.reshape(x_train.shape[0],28,28,1).astype("float")/255.
# x_test = x_test.reshape(x_test.shape[0],28,28,1).astype("float")/255.
x_train = x_train.reshape(x_train.shape[0],28*2,14).astype("float")/255.
x_test = x_test.reshape(x_test.shape[0],28*2,14).astype("float")/255.
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(x_train.shape,x_test.shape)

print(y_train.shape)

def build_model(drop=0.5,optimizer = 'adam'):

    inputs = Input(shape=(28*2,14), name='input')
    x = LSTM(20,name='hidden1')(inputs)
    x = Dropout(drop)(x)

    x = Dense(256, activation='relu',name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation='relu',name='hidden3')(x)
    x = Dropout(drop)(x)
    output1 = Dense(10,name='output')(x)
    model = Model(inputs=inputs,outputs=output1)
    model.compile(optimizer=optimizer,metrics=['acc'],loss="categorical_crossentropy")

    return model

def create_hyper():
    batches = [10,20,30,40,50]
    optimizer = ['rmsprop','adam','adadelta']
    droptout = np.linspace(0.1,0.5,5)
    return{"batch_size" : batches, "optimizer"  : optimizer, "drop": droptout}

model = KerasClassifier(build_fn=build_model, verbose = 1)

hyperparameters = create_hyper()

search = RandomizedSearchCV(estimator=model, param_distributions=hyperparameters,n_iter=10,cv=3,n_jobs=1)
search.fit(x_train,y_train)
pred = search.predict(x_test)
score = accuracy_score(y_test,pred)

# score = search.score(x_test)
print(search.best_params_)
print(score)