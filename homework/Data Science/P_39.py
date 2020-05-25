import random
random.seed(2)
four_random = [random.random() for _ in range(4)]

# print(four_random)

num = list(range(10))
random.shuffle(num)
# print(num)
rannum = random.choice(num)
# print(rannum)

#중복제한 랜덤
lottery = range(60)
ans = random.sample(lottery,10)
# print(ans)


##정규 표현식

import re
re_example = [                          #모두 True
    not re.match("a","cat"),            #'cat'은 'a' 로 시작 안한다
    re.search("a","cat"),
    not re.search("a","dog"),
    3 == len(re.split("[ab]","carbs")),  #a혹은 b을 기준으로 c |a| r |b| s를 나눈다
    "R-D-" == re.sub("[0-9]","-","R2D2")
]

# print(re.split("[as]","vabsdadd"))

list1 = ['a','b','c','d']
list2 = [1,2,3,4]
pa = [pair for pair in zip(list1,list2)]     #[('a',1),('b',2).....]
# print(pa)
letters, nums = zip(*pa)
# print(letters, nums)

def add(a,b): print(a+b)

add(1,2)
try:
    add([1,2])
except TypeError:
    print("add expects two inputs")
add(*[1,2])