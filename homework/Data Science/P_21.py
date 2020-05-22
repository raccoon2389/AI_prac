# P_21.py 함수
def double(x):
    '''
    함수의 설명을 적는다 ex)2를 곱셈 함수
    '''
    return x*2

def f(x):
    return double(x)

y = f(lambda x: x*2) #변수에 함수를 주고싶을 떄 주로 사용한다. 그외에는 잘 사용안함

print(y(2))

def df(i = "Default haha"):
    print(i)
    return 0

df()