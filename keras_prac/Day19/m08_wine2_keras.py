import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense,Dropout
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.decomposition import PCA
from keras.utils import np_utils

e_stop = EarlyStopping(monitor='val_loss', patience=20)
m_check = ModelCheckpoint('./model/{epoch:02d}--{val_loss:.4f}.hdf5',monitor='val_loss',save_best_only=True)

dataset = pd.read_csv('data/winequality-white.csv',sep=';',header=0)

print(dataset.head()) 
# null = dataset.isnull().sum()
# print(null) # 0


x = dataset.loc[:,"fixed acidity":"alcohol"].values

# pca = PCA(n_components=5)
# pca.fit(x)
# x = pca.transform(x)

y = dataset.loc[:,"quality"].values.astype('int')
y = np_utils.to_categorical(y)
y = y[:,3:]

x_train, x_test, y_train, y_test = \
    train_test_split(x,y,test_size=0.2, shuffle=True)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

model = Sequential()
model.add(Dense(20000,activation='relu',input_dim = x_train.shape[1]))
model.add(Dropout(0.2))
model.add(Dense(100,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(30,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(7,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])
model.fit(x_train,y_train,batch_size=5,epochs=100,validation_split=0.25,callbacks=[e_stop,m_check])

loss, acc = model.evaluate(x_test,y_test,batch_size=1)
print(f"loss : {loss} \n acc : {acc}")