import numpy as np


#1 데이터

x = np.array([1,2,3,4])
y = np.array([1,2,3,4])

#2 모델구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(10,input_dim=1,activation='relu'))
model.add(Dense(3))
model.add(Dense(1))
from keras.optimizers import Adam,RMSprop,SGD,Adadelta,Nadam

# optimize = Adam(learning_rate=0.1)
predicted ={}
for lr in np.linspace(0.001,0.1,10):
    pred =[]
    op = [Adam(lr=lr),RMSprop(lr=lr),SGD(lr=lr),Adadelta(lr=lr),Nadam(lr=lr)]
    for optimizer in op:
        print(optimizer)

        model.compile(loss='mse', optimizer=optimizer,metrics=['mse'])

        model.fit(x,y,epochs=100,verbose=0)
        loss = model.evaluate(x,y)

        pred1= model.predict([3.5])
        pred.append(pred1)
    predicted[lr] = pred
print("lr\t\tAdam\t\tRMSprop\t\tSGD\t\tAdadelta\t\tNadam")
for opt in predicted:
    pred = predicted[opt]
    print(f"{lr}\t{pred[0][0][0]}\t{pred[1][0][0]}\t{pred[2][0][0]}\t{pred[3][0][0]}\t{pred[4][0][0]}\t")
