#for
a = {'name': 'Q', 'phone' : '010', 'birth' : '0809'}
for i in a.values():
    print(i)

a = range(10)
for i in a:
    i = i*i
    print(i)
# print('메렁')

#while 조건문 : #참일동안 계속된다.

# if
if 2 : #0 제외 전부 True
    print('True')
else :
    print('False')

#비교연산자 >< == != >= <=

money = 10000
card = 1
if money>= 30000 or card == 1:
    print('한우먹자')
else:
    print('ramen mukja')
jumsu = [90,25,67,80,30]
number = 0
for i in jumsu:
    if i < 60:
        continue
    
    if i >= 60:
        print("경) 합격 (축")
        number = number+1

print('합격인원 : ',number,'명')