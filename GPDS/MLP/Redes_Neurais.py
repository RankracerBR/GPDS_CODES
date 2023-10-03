#Libs
from sklearn.preprocessing import StandardScaler #Normalização de dados
from sklearn.model_selection import train_test_split #Separar os conjuntos de dados
from sklearn.metrics import r2_score #Mede a eficiência do seu modelo
from sklearn.linear_model import SGDRegressor #Regressão linear
from sklearn.neural_network import MLPRegressor # Para implementação da MLP
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


#Variables
#Representando uma camada
Inputs_neoron = [5,6,7]

Weights_neoron = [0.3,5,5.7]

Bias_neuron = 10

Output_neuron = Inputs_neoron[0]*Weights_neoron[0] + Inputs_neoron[1]*Weights_neoron[1] + Inputs_neoron[2]*Weights_neoron[2] + Bias_neuron

print(Output_neuron)
print('\n')

Output_neuron_np = np.dot(Weights_neoron,Inputs_neoron) + Bias_neuron

print(Output_neuron_np)
print('\n')

Inputs_layer = [1,2,3,2.5]

Weights_layer_1 = [0.2,0.8,-0.5,1]
Weights_layer_2 = [0.5,-0.91,0.26,-0.5]
Weights_layer_3 = [-0.26,-0.27,0.17,0.87]

Bias_layer_1 = 2
Bias_layer_2 = 3
Bias_layer_3 = 0.5

Output_layer = [Inputs_layer[0]*Weights_layer_1[0] + Inputs_layer[1]*Weights_layer_1[1] + Inputs_layer[2]*Weights_layer_1[2] + Inputs_layer[3]*Weights_layer_1[3] + Bias_layer_1,
                Inputs_layer[0]*Weights_layer_2[0] + Inputs_layer[1]*Weights_layer_2[1] + Inputs_layer[2]*Weights_layer_2[2] + Inputs_layer[3]*Weights_layer_2[3] + Bias_layer_2,
                Inputs_layer[0]*Weights_layer_3[0] + Inputs_layer[1]*Weights_layer_3[1] + Inputs_layer[2]*Weights_layer_3[2] + Inputs_layer[3]*Weights_layer_3[3] + Bias_layer_3]

print(Output_layer)

Weights_layer = np.array([[0.2,0.8,-0.5,1],
                          [0.5, -0.91, 0.26, -0.5],
                          [-0.26, -0.27, 0.17, 0.87]])

Biases_layer = np.array([[2,3,0.5]])

Output_layer_np = np.dot(Weights_layer, Inputs_layer) + Biases_layer
print(Output_layer_np)
print('\n')

#Construído uma RNA prevendo autonomia a partir do peso do veículo
dataset = pd.read_csv('MLP/auto-mpg.csv')
print(dataset.shape)
print(dataset.head(20))

#Relação entre autonomia e peso
plt.scatter(dataset[["weight"]], dataset[["mpg"]])
plt.xlabel("Peso(lb)")
plt.ylabel("Autonomia(mpg)")
plt.title("Relação entre autonomia e peso")
plt.show()

#Pré-processamento de dados
X = dataset[["weight"]]
X["weight"] = X["weight"] * 0.453592
print(X["weight"])

Y = dataset[["mpg"]]
Y["mpg"] = Y["mpg"] * 0.453592
print(Y["mpg"])
print('\n')

#Normalização dos dados
print(X.describe())
print('\n')

#Realizando a Normalização/Padronização
escala = StandardScaler()
escala.fit(X)

X_norm = X/np.max(X)
dataset["X_norm"] = X_norm
print(dataset["X_norm"].describe())
print('\n')

#Conjuntos de treinamento e de teste
X_norm_train, X_norm_test, Y_train, Y_test = train_test_split(X_norm, Y[["mpg"]], test_size = 0.3)

#Construído a RNA
rna = MLPRegressor(hidden_layer_sizes=(20,8),
                   max_iter=2000,
                   tol = 0.0000001,
                   learning_rate_init = 0.01,
                   solver = "sgd",
                   activation = "logistic",
                   learning_rate = "constant",
                   verbose= 2,)

print(rna.fit(X_norm_train, Y_train))
print('\n')

#Regressão Linear
reg_linear = SGDRegressor(max_iter = 2000, 
                          tol = 0.000001,
                          eta0 = 0.1,
                          learning_rate = "constant",
                          verbose = 2)

print(reg_linear.fit(X_norm_train, Y_train))
print('\n')

#Pós-Processamento
#Previsão do conjunto de teste

Y_rna_previsao = rna.predict(X_norm_test)
Y_reg_linear_previsao = reg_linear.predict(X_norm_test)

#Cáculo de score
r2_rna =  r2_score(Y_test, Y_rna_previsao)
r2_reg_linear = r2_score(Y_test, Y_reg_linear_previsao)

print("Proximidade em relação aos dados reais(MLP): ", r2_rna*100)
print("Proximidade em relação aos dados reais(RL): ", r2_reg_linear*100)

#Gerando os gráficos comparativos
X_test = escala.inverse_transform(X_norm_test)

plt.scatter(X_test, Y_test, alpha= 0.5, label = "Reais")
plt.scatter(X_test, Y_rna_previsao, alpha= 0.5, label = "MLP")
plt.scatter(X_test, Y_reg_linear_previsao, alpha = 0.5, label = "RL")

plt.xlabel("Peso em kg")
plt.ylabel("Autonomia(km/l)")
plt.title("Resultados obtidos: ")
plt.legend(loc=1)
plt.show()