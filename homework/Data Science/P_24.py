########    LIST    #######

x = [0,1,2,3,4,5,6,7,8,9] #[0~9]
zero = x[0] # 0
one = x[1] #1
nine = x[-1] #9
x[0] = -1 #x= [-1,1,2,3,.....,8,9]

first_three = x[:3] # 처음부터 3번째 까지 잘라서 저장한다
three_to_end = x[3:] # 3번쨰부터 끝까지
one_to_f = x[1:4]
without_f_l = x[1:-1] #처음하고 마지막만 제외
copy_of_x = x[:] #전부

every_third = x[::3] # =[-1,3,6,9]
five_to_three = x[5:2:-1] #5번쨰에서 2번째까지 -1순서로 

1 in [1,2,3] #True
0 in [1,2,3] #False

x = [1,2,3]
x.extend([4,5,6]) # [1,2,3,4,5,6]

x = [1,2,3]
y = x + [4,5,6] # y = [1~6]
                # x = [123]

x.append(0) # x=[1230]
y = x[-1] # 0

x,y =[1,2] # x=1 , y =2

_,y = [1,2] # 갯수를 맞춰줘야하므로 빈칸에는 _를 써서 공백임을 나타낸다



######      TUPLE       ######
ilist = [1,2]
ituple = (1,2)

try:
    ituple = (3,4)  # 튜플은 변경 불가능
except TypeError:
    print("튜플은 못바꿈")

def sum_and_product(x,y): # 함수가 복수의 값을 반환할떄 튜플이 유용하다
    return (x+y),(x*y)

sp = sum_and_product(2,3) #sp = (2,3)
s,p = sum_and_product(4,5) # s=9 p = 20

#dictionary
empty_dict = {}
empty_dict2 = dict()
grades = {"Jo":80 ,"rha": 95}

Jo_grade = grades["Jo"] # 80

#sorting

x= [5,34,2,1,4]
y = sorted(x) #정렬된 x 를 리턴
x.sort # x 리스트 자체를 소팅해줌

