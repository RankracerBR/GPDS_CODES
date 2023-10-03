'''Libs'''
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier


'''Functions'''
def generate_columns(db):
    columns = ["pacienteid","classe"]
    exames_column = [f"exame{i}" for i in range ((db.shape[1]-2))]
    columns.extend(exames_column)
    print(columns)
    return columns

'''Variables'''
path = "Pandas_Sklearn/exames.csv"

data_base = pd.read_csv(path)
print(data_base)
print('\n')

print(data_base.head(10))
print('\n')

print(data_base.shape)
print('\n')

columns = generate_columns(data_base)
print(len(columns))
print('\n')

data_base_v2 = pd.DataFrame(data_base.to_numpy(), columns = columns)
data_base_v2.head(10)

print(data_base.isnull())
print('\n')

print(data_base_v2.isnull().sum())
print('\n')

data_base_v3 = data_base_v2.drop(["exame32"], axis = 1)
print(data_base_v3.isnull().sum())

data_base_v3['exame26'] = pd.to_numeric(data_base_v3['exame26'], errors='coerce')

valor_medio = data_base_v3['exame26'].mean()
print(valor_medio)
print('\n')

data_base_v3["exame26"].fillna(valor_medio, inplace = True)

print(data_base_v3.isnull().sum())
print('\n')

data_base_v3.to_csv("dataframe_no_null.csv")

data_base_v3 = data_base_v3.apply(pd.to_numeric, errors='coerce')

data_list = data_base_v3.sum(axis = 1).tolist()

print(data_list)
print('\n')

data_list = [x/32 for x in data_list]
print(data_list)
print('\n')

data_base_v3["exame_medio"] = data_list
print(data_base_v3.head(10))
print('\n')

valores_exames = data_base_v3.drop(columns = ["pacienteid", "classe"])
diagnostico = data_base_v3.classe
seed = 1234
np.random.seed(seed)
treino_x,teste_x,treino_y,teste_y = train_test_split(valores_exames, diagnostico, test_size = 0.25)

print(valores_exames.isnull().sum())
print('\n')

classificador = RandomForestClassifier(n_estimators = 100)
#classificador.fit(treino_x, treino_y)

#previsoes = classificador.predict(teste_x)
#acc = accuracy_score(teste_y,previsoes)*100
#print(acc)

#modelo = DummyClassifier()
#modelo.fit(treino_x,treino_y)
#previsoes = modelo.predict(teste_x)
#acc = accuracy_score(teste_y, previsoes)*100
#print(acc)