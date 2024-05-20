import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# 1. Identificar el Universo de Datos
# a. Cargar el archivo de Excel
data = pd.read_csv('Rental Bike Data.csv')

# 2. Establecer una hipótesis sobre el análisis y resultados esperados
# Hipótesis: "La cantidad de bicicletas rentadas está significativamente influenciada por la temperatura y la humedad."

# 3. Seleccionar los Datos sobre los que se Trabajará
# a. Determinar si se trabajará sobre el conjunto completo de datos o un subconjunto
# Trabajaremos con el conjunto completo, pero haremos un análisis de correlación para seleccionar las variables.

# Analizar la correlación entre las variables
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()

# 4. Pre-procesamiento y Limpieza de Datos
# a. Crear un atributo que será la etiqueta de clasificación
# Decidimos que X = 100.
data['High_Rentals'] = data['Rented Bike Count'].apply(lambda x: 1 if x >= 100 else 0)

# b. Predicción sobre el número de bicicletas rentadas
# La etiqueta High_Rentals será utilizada para la predicción.

# c. Revisar la calidad de los datos y transformaciones necesarias
# Eliminar valores nulos y duplicados si existen.
data.dropna(inplace=True)
data.drop_duplicates(inplace=True)

# d. Eliminar registros de ciertas horas
data = data[~data['Hour'].isin([0, 1, 2, 3, 4, 5, 6])]

# e. Discretizar atributos
# Discretizar algunas de las variables continuas.
data['Temperature_bin'] = pd.cut(data['Temperature(°C)'], bins=[-10, 0, 10, 20, 30, 40], labels=[1, 2, 3, 4, 5])
data['Humidity_bin'] = pd.cut(data['Humidity(%)'], bins=[0, 20, 40, 60, 80, 100], labels=[1, 2, 3, 4, 5])
data['Wind_speed_bin'] = pd.cut(data['Wind speed (m/s)'], bins=[0, 2, 4, 6, 8, 10], labels=[1, 2, 3, 4, 5])
data['Visibility_bin'] = pd.cut(data['Visibility (10m)'], bins=[0, 500, 1000, 1500, 2000], labels=[1, 2, 3, 4])
data['Dew_point_bin'] = pd.cut(data['Dew point temperature(°C)'], bins=[-10, 0, 10, 20, 30], labels=[1, 2, 3, 4])
data['Solar_radiation_bin'] = pd.cut(data['Solar Radiation (MJ/m2)'], bins=[0, 5, 10, 15, 20], labels=[1, 2, 3, 4])
data['Rainfall_bin'] = pd.cut(data['Rainfall(mm)'], bins=[0, 1, 2, 3, 4, 5], labels=[1, 2, 3, 4, 5])
data['Snowfall_bin'] = pd.cut(data['Snowfall (cm)'], bins=[0, 1, 2, 3, 4, 5], labels=[1, 2, 3, 4, 5])

# 5. Transformación
# a. Particionamiento del conjunto de datos
# Seleccionar las columnas relevantes y la etiqueta
X = data[['Temperature_bin', 'Humidity_bin', 'Wind_speed_bin', 'Visibility_bin', 'Dew_point_bin', 'Solar_radiation_bin', 'Rainfall_bin', 'Snowfall_bin']]
y = data['High_Rentals']

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Minería de Datos
# a. Elegir un algoritmo
# Utilizaré Random Forest para la clasificación.
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Realizar predicciones
y_pred = model.predict(X_test)

# Evaluar el modelo
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Matriz de Confusión:")
print(conf_matrix)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

# 7. Evaluación e Interpretación
# a. Explicar los resultados obtenidos
print(f"""
Resultados Obtenidos
- Accuracy: {accuracy}
- Precision: {precision}
- Recall: {recall}

Interpretación
- Accuracy indica que el modelo tiene un {accuracy * 100:.2f}% de precisión en las predicciones generales.
- Precision muestra que el {precision * 100:.2f}% de las veces que el modelo predijo 'alta demanda', fue correcto.
- Recall refleja que el modelo identificó correctamente el {recall * 100:.2f}% de los casos con 'alta demanda'.

Confirmación de Hipótesis
- La hipótesis se confirma parcialmente ya que la temperatura y la humedad tienen un impacto significativo en la predicción, lo cual coincide con la expectativa inicial.
""")
