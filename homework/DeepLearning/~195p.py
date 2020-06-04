#흰초 책 chapter 04~09 일요일00시 
import numpy as np
n=np.array([1,2,3,4])
type(n) #ndarray

storage = [1,2,3,4]
new_storage =[]

for n in storage:
    n += n
    new_storage.append(n)
# print(new_storage) #1,2,4,6,8

arr = np.array([1,2,3,4])

# print(arr+arr)#,1,4,6,8
# print(arr-arr)#0,0,0,0
# print(1/arr)#1,0.5,00.3333,0.25

#p191
# print(arr[0:1]) #[1]
arr[1:4]=24
# print(arr[1:4])

arr1 = np.array([1,2,3,4,5])
arr2 = arr1
arr2[0]=100
# print(arr1) # 100,2,3,4,5
arr1 = np.array([1,2,3,4,5])
arr2 = arr1.copy()
# print(arr1) # 1,2,3,4,5

arr_list = [i for i in range(10)]
print('list')
print('arr_list : ',arr_list)
print()

arr_list_copy = arr_list[:]
arr_list[0]=100 # List를 슬라이스한 카피는 배열 주소가 아닌 배열 자체를 반환하므로 값이 변하지 않음
print('copy : ', arr_list_copy)

arr_np = np.arange(10)
print('numpy')
arr_np2 = arr_np[0:4] # 슬라이싱 해도 view를 반환한다
arr_np2[0]=100
print(arr_np)

arr_np3 = arr_np.copy() # copy를 해줘야 변하지 않는다
arr_np3[0]=10000
print(arr_np)

