import numpy as np

arr = np.array ([2,4,6,7])
# print(arr[np.array([True,False,True,True])]) # True False로 인덱싱 가능

# print(arr[arr%3==1])

# print(arr[arr%2==0])


arr = np.array([4,-9,-10,10,50])
# print(np.abs(arr))
# print(np.sqrt(np.exp(arr)))

#1차원 배열만을 대상으로 하는 집합함수 가 있다
# np.unique() 중복제거하고 정렬한 결과 
# np.union1d(x,y) x와y의 합집합을 반환
# np.intersect1d(x,y) 교집합 반환
# np.setdiff1d(x,y) x에서 y를 뺀 차집합
# arr = np.array([2,5,7,9,5,2])
# arr1 = np.array([2,5,8,3,1])
arr = [2,5,7,9,5,2]
arr1 = [2,5,8,3,1]
# print(arr,arr1)
# print(np.unique(arr,arr1))
# print(np.union1d(arr,arr1))
# print(np.intersect1d(arr,arr1))
# print(np.setdiff1d(arr,arr1))

# np.random.rand() 0이상 1미만의 랜덤한 수
# np.random.randint(x,y,z) x이상 y미만 정수를 z 개 생성 z에 (2,3)등으로 행렬만들수있다
# np.random.normal() 가우스 분포 따르는 난수

from numpy.random import randint

arr = randint(0,10,(5,2))
arr1 = randint(0,1,3)
# print(arr,arr1)

arr = arr.reshape(-1,2)
# print(arr)
# print(arr.shape)
# print(arr.reshape(2,-1))

# print(arr[1:,1])
# print(arr[1,1])

arr = np.arange(1,10).reshape(3,3)
# print(arr[0,2],arr[0,:])

#axis= 0 세로마다 처리 axis=1 가로마다 처리
# print(arr.sum(),arr.sum(axis=0),arr.sum(axis=1))

#팬시 인덱싱 arr[[1,2,0,]] 1행 2행 0행으로 새로운 배열 만든다.
# print(arr[[1,2,0]]) 

arr = np.arange(9).reshape(3,3)
# print(arr.T)

# arr= np.array([15,30,5])
# print(arr.argsort())
# print(np.sort(arr))
# print(arr.sort(0))
# print(arr)



# print(np.dot(arr,arr))
# arr = arr.reshape(9)
# print(np.linalg.norm(arr))

# print(arr.mean(axis=0), arr.sum(axis=1), arr.min(),arr.argmax())
# print(arr+1)
broad = np.array([1,2,3])
# print(arr-broad)

arr = randint(0,30,15).reshape(5,3)
arr = arr.T
arr1 = arr[:,[1,2,3]]
print(arr1)
arr1.sort(0)
print(arr1)

np.random.seed(0)

def make_image(m,n):
    img = randint(0,5,(n,m))
    return img

def chage_little(matrix):
    shap = matrix.shape
    mask = randint(0,2,shap).astype('bool')
    print(mask)
    matrix[mask]= randint(0,5,matrix[mask].shape)
    return matrix

img1 = make_image(3,3)
print(img1)
img2 = chage_little(img1)
print(img2)
img3 = img2-img1
img3 = np.abs(img3)