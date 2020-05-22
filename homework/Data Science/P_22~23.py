# P_22.py
# string
# '' 와 "" 둘다 사용 가능하지만 앞뒤는 같은것을 사용해주어야한다
single = 'data'
double = "data"
tab = "\t"
print(len(tab)) #1

not_tab = r"\t" # 문자열 앞에 r(raw)를 붙혀두면 역슬래쉬의 특수 문법을 무시하고 그대로 내보낸다 (ex> "d:use\dsds\")

print(len(not_tab))#2

#f-string
first = "roy"
last = "hong"

full1 = first + '' + last # 문자열 합치기
full2 = "{0} {1}".format(first,last) # format을 이용해서 붙히기

full3 = f"{first} {last}" # f-string 을 이용하여 합치기

#try와 except
try:
    print(0/0)          #try안을 실행해보고 안되면 except 로 넘어감

except ZeroDivisionError:
    print("cannot 0 division") # 0을 못나눠서 error로 실행이 멈추는 대신 이 문장이 출력됨

finally:
    pass

i_list = [1,2,3]
h_list = ["string", 0.1, True]
list_list = [i_list,h_list,[]]

list_len = len(i_list) #3
list_sum = sum(i_list) #6

print(list_len,list_sum)
