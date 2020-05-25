class counter :
    """주석달수 있다 끼얏호우"""
    def __init__(self, count=0):
        self.count = count
    def __repr__(self):
        return f"counter(count={self.count})"
    def click(self,times):
        self.count += times
    def read(self):
        return f"counter(count={self.count})"
    def reset(self):
        self.count = 0


class Nocounter(counter):
    #counter와 동일한 것 출력

    def reset(self):
        pass

# cl1=counter()


# print(cl1.read())

def generate_range(n):
    i=0
    while i <n:
        yield i # yield가 호출될때마다 제너레이터에 해당 값을 생성하기 때문에 함수 재사용은 힘들다
        i +=1

# for i in generate_range(10): 
#     print(f"i:{i}")

even_b = [i for i in generate_range(20) if i %2==0]
print(even_b)
names = ["Als","BB","CC","DD"]
for i, name in enumerate(names):
    print(f"name {i}, is {name}")