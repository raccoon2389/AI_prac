import numpy as np
from keras.models import Sequential
from keras.layers import Dense,LSTM

#1 데이터

a = np.array(range(1,11))
size = 5

def split_X(seq,size):
    aaa = []
    for i in range(len(seq) - size +1 ): # len(seq) - size +1 = 몇개의 행을 갖을수 있는지 계산
        subset = seq[i: (i+size)] # 한행에 넣을 데이터 추출
        aaa.append(subset) # subset에 있는 item을 shape에 맞게 aaa 뒤에 행 추가
    print(type(aaa))
    return np.array(aaa)


dataset = split_X(a,size)

print('===============================')
print(dataset)
print('===============================')
