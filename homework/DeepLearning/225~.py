import pandas as pd

fruits = {"orange" : 2,"banana" : 3}
# print(pd.Series(fruits))

data = {"fruits":["apple","orange","banana","strawberry","kwiwi"],
        "year":[2001,2002,2001,2008,2077],
        "time":[1,4,5,6,3]}

df = pd.DataFrame(data)
# print(df)

# series : pandas에서 1차원 배열
index = ["ap","ba","ca"]
data =[10,5,8]

seri = pd.Series(data,index=index)
# print(seri)


# 참조
# print(seri[0:1])
# print(seri["ca"])
item1 = seri[0:2]
item2 = seri[["ap","ba"]]
# print(item1, item2)

# print(seri.values)
# print(seri.index)
seri = seri.append(pd.Series({"da" : 200}))
# print(seri)
seri.drop("ca")
# print(seri)
mask = [True,False,True,False]
# print(seri[mask])
# print(seri[seri>9])

# print(seri.sort_index())
# print(seri.sort_values(ascending=False))

df = pd.DataFrame([seri,seri,seri])

df.index = ["a","b","c"]
print(df)
