def doubler(f):
    def g(x):
        return 2*f(x)

    return g # 2개 이상의 인자를 받는경우 문제가 생긴다

def f1(x):
    return x+1

g = doubler(f1)
assert g(3)==8, "error1"
assert g(-1)==0, "error2"

def f2(x,y):
    return x+y

g = doubler(f2)

try:
    g(1,2)
except TypeError:
    print("error3")

def magicc(*arg, **kwargs):
    print("unnamed args: ", arg)
    print("keyword args: ", kwargs)

magicc(1,2,key='word',key2 ='word2')

def other_magic(x,y,z):
    return x+y+z

x_y_list = [1,2]
z_dict = {"z":3}
assert other_magic(*x_y_list,**z_dict)== 6,'error4'

def doubler_correct(f):
    def g(*args,**kwargs):
        return 2 * f(*args,**kwargs)
    return g

g = doubler_correct(f2)
print(g(2,2)) 