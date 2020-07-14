import sqlite3

conn = sqlite3.connect("test.db")

cursor = conn.cursor()

cursor.execute("""CREATE TABLE IF NOT EXISTS supermarket(Itemno INTEGER, Category TEXT,FoodName TEXT, Company TEXT, Price INTEGER)""")

sql = "DELETE FROM supermarket"
cursor.execute(sql)

sql = "INSERT into supermarket(Itemno,Category,FoodName,Company,Price) values (1,'과일','자몽','마트',1500)"
cursor.execute(sql)

sql = "INSERT into supermarket(Itemno,Category,FoodName,Company,Price) values (2,'음료수','망고주스','편의점',1000)"
cursor.execute(sql)

sql = "INSERT into supermarket(Itemno,Category,FoodName,Company,Price) values (3,'고기','소고기','하나로마트',10000)"
cursor.execute(sql)

sql = "INSERT into supermarket(Itemno,Category,FoodName,Company,Price) values (3,'고기','소고기','하나로마트',10000)"
cursor.execute(sql)

sql = "SELECT * FROM supermarket"

cursor.execute(sql)

rows = cursor.fetchall()

for row in rows:
    print(str(row[0])+" "+str(row[1])+" " +
        str(row[3])+" "+str(row[4]))
conn.commit()
conn.close()
