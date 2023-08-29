import pandas as pd
import numpy as np

# Seleccionamos nuestros datos con los cuales vamos a trabajar, este es un dataframe de un banco con caracteristicas importantes
# de sus clientes como edad, profesión, crédito, estatus social, entre otras.
data = 'dataset.csv'

try:
    dataset = pd.read_csv("C:/Users/luisl/Documents/Módulo de Uresti/dataset.csv")
    # Realiza operaciones con el DataFrame 'df'
except FileNotFoundError:
    print(f"El archivo 'C:/Users/luisl/Documents/Módulo de Uresti/dataset.csv' no fue encontrado.")

# En este paso vamos a eliminar los datos faltantes, en nuestro dataset los datos faltantes estan expresados por 'unknown', para ello los remplazamos
# por NA para después aplicar una función que borre todas las filas que tengan algún dato faltante en alguna de nuestras columnas, esto con el fin
# de poder tener nuestra base de datos limpia para un mejor modelo
dataset.replace('unknown', pd.NA, inplace=True)
data_limpia = dataset[dataset != '<NA>'].dropna()

# Tenemos la variable edad la cual esta representada por varias edad por lo cuál se procedio a categorizarla por rango de edades, 
# en este caso por 5 categorías
intervalos = [0, 18, 40, 60, 80, 100]
labels = ['0-18', '19-40', '41-60', '61-80','81-100']
data_limpia['age'] = pd.cut(data_limpia['age'], bins=intervalos, labels=labels, right=False)

# Ahora nos fijamos en nuestra variable a predecir en nuestro dataset, en este caso vamos a predecir de acuerdo a distintas variables de nuestro
# dataset si un cliente del banco tiene un prestamo o no para ello nos fijamos en los valores que possee la columna loan que es "yes" o "no"
data_limpia['loan'].value_counts()

# Ahora escogemos algunos valores para implementar nuestro modelo, en este caso ocuparemos los de la columna "housing" ya que son los datos
# que tienen más variabilidad con respecto a las otras variables
si_samples = data_limpia[data_limpia['housing'] == 'yes'].sample(n=4000)
no_samples = data_limpia[data_limpia['housing'] == 'no'].sample(n=2000)

# Concadenamos los valores seleccionados
datos_seleccionados = pd.concat([si_samples, no_samples])
# Déspues ponemos los datos aleatorios para que no esten ordenados primeros lo de "yes" y "no"
datos_seleccionados = datos_seleccionados.sample(frac=1, random_state=42)

# Iniciamos con la implementación del clasificador Naive Bayes
def clasificador_naive_bayes(X_train, y_train, entrada):
    clases = np.unique(y_train)
    total_de_clases = {cls: np.count_nonzero(y_train == cls) for cls in clases}
    
    Proba_condi = {}
    for cls in clases:
        Proba_condi[cls] = {}
        for columna in range(X_train.shape[1]):
            Proba_condi[cls][columna] = {}
            unique_vals, counts = np.unique(X_train[y_train == cls, columna], return_counts=True)
            Proba_condi[cls][columna] = dict(zip(unique_vals, counts))
    
    def calcular_probabilidades(entrada, clase_real):
        probabilidades = np.empty(len(clases))
        for i, cls in enumerate(clases):
            probabilidad_previa = total_de_clases[cls] / (len(y_train) + len(clases))
            producto = probabilidad_previa
            for columna, valor_atributo in enumerate(entrada):
                probabilidad_condicional_del_valor = Proba_condi[cls][columna].get(valor_atributo, 0) / total_de_clases[cls]
                producto *= probabilidad_condicional_del_valor
            probabilidades[i] = producto
        return probabilidades
    
    total_casos = len(y_val)
    correctos = 0
    erroneos = 0

    for i in range(total_casos):
        entrada = X_val[i]
        clase_real = y_val[i]
        probabilidades = calcular_probabilidades(entrada, clase_real)
        prediccion_idx = np.argmax(probabilidades)
        prediccion_clase = clases[prediccion_idx]

        #print('Valor real: ', y_val[i])
        #print('Valor predecido: ', prediccion_clase)

        if y_val[i] == prediccion_clase:
            correctos += 1
        else:
            erroneos += 1

        #print('Total de casos: ', total_casos - i)
        #print('Valor predecido: ', prediccion_clase)
        #print('Iteración: ', i)

    print('Total de casos: ', total_casos)
    print('Correctos: ', correctos)
    print('Erroneos: ', erroneos)
    print('Accuracy: ', correctos / total_casos)

datos_seleccionados.reset_index(inplace=True, drop=True)
datos_seleccionados['loan'].value_counts()
y = datos_seleccionados['loan'].values 
X = datos_seleccionados[['age', 'job', 'marital', 'housing', 'contact',
                         'month', 'day_of_week',
                         'previous']].values 

#y_train = y[:4500]
#y_val = y[4500:]

#X_train = X[:4500]
#X_val = X[4500:]
total_samples = len(y)
random_indices = np.random.permutation(total_samples)
split_index = int(total_samples * 0.8)  # 90% para entrenamiento, 10% para validación

y_train = y[random_indices[:split_index]]
y_val = y[random_indices[split_index:]]

# Generar datos aleatorios para X_train y X_val
X_random_indices = np.random.permutation(total_samples)
X_train = X[X_random_indices[:split_index]]
X_val = X[X_random_indices[split_index:]]
print("Prueba ------------------------------ 1")
clasificador_naive_bayes(X_train, y_train, X_val)

total_samples = len(y)
random_indices = np.random.permutation(total_samples)
split_index = int(total_samples * 0.8)  # 90% para entrenamiento, 10% para validación

y_train = y[random_indices[:split_index]]
y_val = y[random_indices[split_index:]]

# Generar datos aleatorios para X_train y X_val
X_random_indices = np.random.permutation(total_samples)
X_train = X[X_random_indices[:split_index]]
X_val = X[X_random_indices[split_index:]]

print("Prueba ------------------------------ 2")
clasificador_naive_bayes(X_train, y_train, X_val)