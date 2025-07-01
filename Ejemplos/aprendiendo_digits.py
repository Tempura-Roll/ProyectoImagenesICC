import pandas as pd
from sklearn import datasets

losDigitos = datasets.load_digits()
print("\nLa primera imagen es:")
print(losDigitos["images"][0])

print("\nLa cantidad total de imágenes es:")
print(len(losDigitos["images"]))

print("\nLos targets son:")
print(losDigitos["target"])

print(losDigitos)

#hacemos un dataframe con la imagen 1 (índice 0)
df1 = pd.DataFrame(data=losDigitos["images"][0])
print(df1)

#ahora hacemos un separador
separador = pd.DataFrame(data=[[0,0,0,0,0,0,0,0]])

#ahora hacemos un dataframe con la segunda imagen (índice 1)
df2 = pd.DataFrame(data=losDigitos["images"][1])

df1 = pd.concat([df1,separador,df2], ignore_index=True)

print("\nAsí va quedando mi dataframe:")
print(df1)

df1 = pd.concat([df1, separador], ignore_index=True)
#iterativamente, agregaremos a las siguientes imagenes
for i in range(2,1797):
    df2 = pd.DataFrame(data=losDigitos["images"][i])
    df1 = pd.concat([df1, df2, separador], ignore_index=True)


print(df1)

#df1.to_csv("Matriz_Digits.csv",index=False)

#Matrices Aplanadas

df3 = pd.DataFrame(data=losDigitos["data"][0])

separador2 = pd.DataFrame(data=[["X"]])

#ahora hacemos un dataframe con la segunda imagen (índice 1)
df4 = pd.DataFrame(data=losDigitos["data"][1])

df3 = pd.concat([df3,separador2,df4], ignore_index=True)

df3 = pd.concat([df3, separador2], ignore_index=True)

for i in range(2,1797):
    df4 = pd.DataFrame(data=losDigitos["data"][i])
    df3 = pd.concat([df3,df4,separador2],ignore_index=True)

df3.to_csv("Matrices_Aplandas.csv",index=False)