#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Manejo de datos
import pandas as pd
import numpy as np
import random

# Preprocesamiento e imputación (habilita IterativeImputer experimental)
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder
)

# Reducción de dimensionalidad
from sklearn.decomposition import PCA

# Modelado y validación
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

# Modelos y balanceo
import xgboost as xgb
from imblearn.over_sampling import SMOTENC

# Métricas y evaluación
from sklearn.metrics import (
    make_scorer,
    f1_score,
    recall_score,
    precision_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve
)

# Análisis estadístico y diagnóstico
from scipy.stats import spearmanr, chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.inspection import permutation_importance

# Interpretabilidad
import shap

# Visualización
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns


# Mostrar todas las columnas al imprimir DataFrames
pd.set_option('display.max_columns', None)

# Silenciar advertencias no críticas para mantener el output limpio
import warnings
warnings.filterwarnings("ignore")


# A partir de los resultados de nuestro Análisis Exploratorio de Datos (EDA), identificamos varias características que tienen un impacto significativo en la deserción (churn), como el tipo de servicio de Internet, la antigüedad del cliente (Tenure) y otras.
# Sin embargo, también existen variables que parecen estar correlacionadas, pero que en realidad pueden estar influenciadas por otros factores.
# En esta sección, reexaminaremos todas las características y aplicaremos diversas técnicas de ingeniería de características y preprocesamiento antes de construir el modelo de aprendizaje automático.

# In[2]:


# Cargar el archivo CSV con los datos originales
datos_telco = pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Convertir la columna 'TotalCharges' a numérica y reemplazar valores no válidos por NaN
datos_telco['TotalCharges'] = pd.to_numeric(datos_telco['TotalCharges'], errors='coerce')


# In[3]:


# Crear una tabla de contingencia entre las variables 'MultipleLines' y 'Churn'
tabla_frecuencias = pd.crosstab(datos_telco['MultipleLines'], datos_telco['Churn'])

# Mostrar la tabla resultante
tabla_frecuencias


# In[4]:


# Mostrar información general del DataFrame
datos_telco.info()


# In[5]:


# Filtrar los registros donde la columna 'TotalCharges' tiene valores faltantes
datos_telco[datos_telco['TotalCharges'].isna()]


# Imputaremos los valores faltantes en TotalCharges con 0, ya que estos corresponden a clientes nuevos cuyos cargos totales aún no se han acumulado.

# In[6]:


# Reemplazar los valores faltantes en 'TotalCharges' por cero
datos_telco['TotalCharges'] = datos_telco['TotalCharges'].fillna(0)


# In[7]:


# Convertir la columna 'TotalCharges' a tipo numérico y forzar errores a NaN
datos_telco['TotalCharges'] = pd.to_numeric(datos_telco['TotalCharges'], errors='coerce')

# Identificar columnas categóricas y numéricas
columnas_categoricas = datos_telco.select_dtypes(include=['object']).columns.tolist()
columnas_numericas = datos_telco.select_dtypes(exclude=['object']).columns.tolist()

# Variable categórica que requiere codificación especial
columnas_codificadas = ['SeniorCitizen']

# Excluir columnas no deseadas del conjunto categórico
excluir_cat = ['customerID', 'Churn']
columnas_cat_final = [col for col in columnas_categoricas if col not in excluir_cat]

# Excluir columnas no deseadas del conjunto numérico
excluir_num = ['SeniorCitizen']
columnas_num_final = [col for col in columnas_numericas if col not in excluir_num]

# Mostrar los resultados
print("Columnas categóricas:", columnas_cat_final, '\n')
print("Columnas numéricas:", columnas_num_final, '\n')
print("Columnas categóricas codificadas:", columnas_codificadas, '\n')


# In[8]:


# Crear una copia del DataFrame original para trabajar sin modificar los datos base
datos_telco_copia = datos_telco.copy()


# In[9]:


# Definir la variable objetivo
objetivo = 'Churn'

# Mapear la columna 'Churn' a valores numéricos: No → 0, Yes → 1
datos_telco_copia[objetivo] = datos_telco['Churn'].map({'No': 0, 'Yes': 1})


# In[10]:


# Función para calcular la correlación de Spearman entre variables numéricas y la variable objetivo
def correlacion_spearman(df_datos, columnas_numericas, columna_objetivo):
    resultados = []
    for col in columnas_numericas:
        correlacion, _ = spearmanr(df_datos[col], df_datos[columna_objetivo])
        resultados.append({
            'Variable': col,
            'Correlacion_Spearman': correlacion,
            'Correlacion_Absoluta': abs(correlacion)
        })
    
    # Retornar un DataFrame ordenado de mayor a menor correlación absoluta
    return (
        pd.DataFrame(resultados)
        .sort_values(by='Correlacion_Absoluta', ascending=False)
        .reset_index(drop=True)
    )

# Calcular las correlaciones para las variables numéricas
correlaciones_numericas = correlacion_spearman(
    df_datos=datos_telco_copia,
    columnas_numericas=columnas_num_final,
    columna_objetivo=objetivo
)

# Mostrar los resultados
correlaciones_numericas


# Para las variables numéricas, concluimos que la antigüedad (tenure) es la característica más correlacionada con la deserción (churn).
# Sin embargo, durante el análisis exploratorio de datos (EDA), también encontramos que otras variables, como Total Charges y Monthly Charges, están correlacionadas con la antigüedad.
# Para abordar este problema, podemos realizar una verificación de multicolinealidad utilizando el Factor de Inflación de la Varianza (VIF, por sus siglas en inglés).

# In[11]:


# Conservar únicamente las columnas numéricas y asegurar que sean de tipo numérico
datos_telco_copia = datos_telco_copia[columnas_num_final].apply(pd.to_numeric, errors='coerce')

# Reemplazar valores infinitos por NaN y eliminar filas con valores faltantes
datos_telco_copia = datos_telco_copia.replace([np.inf, -np.inf], np.nan).dropna()


# In[12]:


# Función para calcular el Factor de Inflación de la Varianza (VIF)
def calcular_vif(dataframe):
    tabla_vif = pd.DataFrame()
    tabla_vif["Variable"] = dataframe.columns
    tabla_vif["VIF"] = [variance_inflation_factor(dataframe.values, i) for i in range(dataframe.shape[1])]
    
    # Clasificar el nivel de colinealidad
    tabla_vif["Nivel"] = tabla_vif["VIF"].apply(
        lambda v: "Bajo" if v < 5 
        else "Medio" if v < 10 
        else "Alto"
    )
    
    # Retornar la tabla ordenada de mayor a menor VIF
    return (
        tabla_vif.sort_values(by='VIF', ascending=False)
                 .reset_index(drop=True)
    )

# Calcular los valores VIF para las variables numéricas seleccionadas
calcular_vif(datos_telco_copia)


# A partir de los resultados del VIF, podemos observar que Total Charges y Tenure presentan un nivel moderado de multicolinealidad, mientras que Monthly Charges muestra un nivel relativamente bajo. Un nivel moderado indica que existe cierta correlación entre estas variables, lo cual debe tenerse en cuenta al construir el modelo de aprendizaje automático. Considerar este factor nos permite desarrollar un modelo más eficiente y robusto.
# 
# En la etapa de modelado, compararemos los resultados de evaluación entre usar todas las variables y usar un conjunto reducido de variables, ya sea eliminando una de las más colineales o aplicando una técnica de reducción de dimensionalidad (PCA).
# 
# Desde una perspectiva matemática, Total Charges puede derivarse esencialmente de la multiplicación entre Tenure y Monthly Charges, ajustada por posibles descuentos. Por lo tanto, podemos eliminar Total Charges, ya que su información ya está representada por las otras dos variables.

# In[13]:


# Crear una lista de variables numéricas excluyendo 'TotalCharges'
columnas_num_sin_total = [col for col in columnas_num_final if col != 'TotalCharges']

# Calcular el VIF solo para las variables seleccionadas
calcular_vif(datos_telco_copia[columnas_num_sin_total])


# Después de eliminar la columna Total Charges, podemos observar que el nivel de multicolinealidad se ha reducido significativamente.

# In[14]:


# Seleccionar las columnas numéricas
datos_numericos = datos_telco_copia[columnas_num_final]

# Escalar los datos con MinMaxScaler
escalador = MinMaxScaler()
datos_escalados = escalador.fit_transform(datos_numericos)

# Aplicar PCA sin limitar el número de componentes
modelo_pca = PCA()
modelo_pca.fit(datos_escalados)

# Calcular la varianza explicada y su acumulado
varianza_explicada = modelo_pca.explained_variance_ratio_
varianza_acumulada = np.cumsum(varianza_explicada)


# In[15]:


# Definir el número de componentes principales a analizar
n_componentes = 2

# Obtener la varianza acumulada hasta ese componente
varianza_acumulada_n = varianza_acumulada[n_componentes - 1]

# Mostrar el resultado formateado
print(f"Varianza acumulada con {n_componentes} componentes: {varianza_acumulada_n:.2f}")


# A partir de los resultados del PCA, podemos observar que con solo dos dimensiones ya es posible representar casi toda la información de los datos numéricos, explicando aproximadamente el 98 % de la varianza.

# In[16]:


# Aplicar PCA seleccionando dos componentes principales
modelo_pca_2 = PCA(n_components=2)
datos_pca = modelo_pca_2.fit_transform(datos_escalados)

# Crear un DataFrame con los resultados del PCA
df_componentes_pca = pd.DataFrame(
    datos_pca,
    columns=['Componente_1', 'Componente_2']
)

# Mostrar las dos primeras filas del DataFrame
df_componentes_pca.head(2)


# In[17]:


# Calcular el VIF (Factor de Inflación de la Varianza) sobre los componentes principales del PCA
calcular_vif(df_componentes_pca)


# Recomendaría aplicar el Análisis de Componentes Principales (PCA) en lugar de eliminar la variable TotalCharges. La razón es que el PCA permite conservar la información de las tres variables (tenure, MonthlyCharges y TotalCharges) transformándolas en nuevos componentes principales. Aunque la interpretabilidad de estos componentes es menos intuitiva en comparación con simplemente descartar una variable, el PCA es estadísticamente más robusto, ya que mitiga la multicolinealidad sin sacrificar la información subyacente contenida en las variables originales.

# In[18]:


# Función para calcular el coeficiente V de Cramér entre dos variables categóricas
def cramers_v(var1, var2):
    # Crear la tabla de contingencia
    tabla_contingencia = pd.crosstab(var1, var2)
    
    # Calcular el estadístico chi-cuadrado
    chi2 = chi2_contingency(tabla_contingencia)[0]
    
    # Calcular el número total de observaciones
    n = tabla_contingencia.to_numpy().sum()
    
    # Calcular la dimensión mínima (ajuste de corrección)
    min_dim = min(tabla_contingencia.shape) - 1
    
    # Calcular el valor de Cramér V
    v_cramer = np.sqrt(chi2 / (n * min_dim))
    return v_cramer


# Función para calcular el V de Cramér entre las variables categóricas y la variable objetivo
def correlacion_categorica(df_datos, columnas_categoricas, columna_objetivo):
    resultados = []
    for col in columnas_categoricas:
        v = cramers_v(df_datos[col], df_datos[columna_objetivo])
        resultados.append({'Variable': col, 'Cramers_V': v})
    
    # Retornar un DataFrame ordenado por correlación descendente
    return (
        pd.DataFrame(resultados)
        .sort_values('Cramers_V', ascending=False)
        .reset_index(drop=True)
    )

# Calcular las correlaciones categóricas respecto a la variable objetivo
correlaciones_categoricas = correlacion_categorica(datos_telco, columnas_cat_final, objetivo)

# Mostrar resultados
correlaciones_categoricas


# In[19]:


# Calcular el coeficiente de Cramér V para las variables categóricas codificadas
correlaciones_codificadas = correlacion_categorica(
    datos_telco,
    columnas_codificadas,
    objetivo
)

# Mostrar los resultados
correlaciones_codificadas


# Ahora combinaremos todos los resultados de correlación anteriores en un nuevo DataFrame para identificar qué variables presentan las correlaciones más altas.

# In[20]:


# Renombrar las columnas de correlación para unificarlas
df_corr_num = correlaciones_numericas.rename(columns={"Correlacion_Absoluta": "Correlacion"})[["Variable", "Correlacion"]]
df_corr_cat = correlaciones_categoricas.rename(columns={"Cramers_V": "Correlacion"})[["Variable", "Correlacion"]]
df_corr_cod = correlaciones_codificadas.rename(columns={"Cramers_V": "Correlacion"})[["Variable", "Correlacion"]]

# Combinar todas las correlaciones en un solo DataFrame
df_correlaciones_total = pd.concat([df_corr_num, df_corr_cat, df_corr_cod], ignore_index=True)

# Ordenar las variables por mayor nivel de correlación
df_correlaciones_total = (
    df_correlaciones_total.sort_values(by='Correlacion', ascending=False)
                          .reset_index(drop=True)
)

# Mostrar el resultado final
df_correlaciones_total


# Las cinco variables más fuertemente asociadas con la deserción (churn) son Contract, Tenure, Online Security, Tech Support e Internet Service. Entre ellas, Contract presenta la correlación más alta con churn, lo que indica que el tipo de contrato desempeña un papel fundamental en la retención de clientes. Tenure resalta la importancia del ciclo de vida del cliente, mientras que los servicios de valor agregado como Online Security y Tech Support contribuyen adicionalmente a reducir el riesgo de deserción. Por su parte, el tipo de Internet Service también se perfila como un factor clave que influye en el comportamiento del cliente.
# 
# A continuación, nos centraremos en las variables con una fuerza de correlación superior a 0.15. Se establece este umbral porque algunas variables, como Senior Citizen, Dependents y Partner, pueden ejercer una influencia significativa sobre la deserción, incluso si sus correlaciones son relativamente pequeñas.

# In[21]:


# Mostrar las 10 variables numéricas con mayor correlación absoluta respecto a la variable objetivo
correlaciones_numericas.sort_values("Correlacion_Absoluta", ascending=False).head(10)


# In[22]:


# Seleccionar variables numéricas con correlación absoluta mayor a 0.15
variables_numericas_sel = correlaciones_numericas[
    correlaciones_numericas["Correlacion_Absoluta"] > 0.15
]["Variable"].tolist()

# Seleccionar variables categóricas con valor de Cramér V mayor a 0.15
variables_categoricas_sel = correlaciones_categoricas[
    correlaciones_categoricas["Cramers_V"] > 0.15
]["Variable"].tolist()

# Incluir las variables categóricas codificadas
variables_categ_cod_sel = correlaciones_codificadas["Variable"].tolist()


# In[23]:


# Combinar todas las variables seleccionadas (numéricas, categóricas y categóricas codificadas)
todas_las_variables = (
    variables_numericas_sel +
    variables_categoricas_sel +
    variables_categ_cod_sel
)

# Mostrar la lista final de variables seleccionadas
todas_las_variables


# In[24]:


# Convertir la variable objetivo 'Churn' a valores numéricos (No → 0, Yes → 1)
datos_telco['Churn'] = datos_telco['Churn'].map({'No': 0, 'Yes': 1}).astype('int8')


# In[25]:


# Definir las variables predictoras (X) y la variable objetivo (y)
X = datos_telco[todas_las_variables]
y = datos_telco['Churn']

# Dividir los datos en conjuntos de entrenamiento y prueba (80% - 20%)
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# A continuación, crearemos dos preprocesadores: uno que incluya PCA y otro que no lo utilice.
# 
# Este enfoque nos permitirá evaluar si el PCA reduce la multicolinealidad y analizar su impacto en las métricas de desempeño, como la precisión (accuracy) y la sensibilidad (recall).

# In[26]:


# Pipeline para variables numéricas: imputación iterativa + normalización
pipeline_numerico = Pipeline([
    ('imputador', IterativeImputer(random_state=0, min_value=0)),
    ('escalador', MinMaxScaler())
])

# Pipeline para variables categóricas: imputación del valor más frecuente + codificación one-hot
pipeline_categorico = Pipeline([
    ('imputador', SimpleImputer(strategy='most_frequent')),
    ('codificador', OneHotEncoder(handle_unknown='ignore'))
])

# Pipeline para variables categóricas codificadas (solo imputación)
pipeline_categ_cod = Pipeline([
    ('imputador', SimpleImputer(strategy='most_frequent'))
])

# Combinación de todos los pipelines en un preprocesador general
preprocesador_1 = ColumnTransformer([
    ('numerico', pipeline_numerico, variables_numericas_sel),
    ('categorico', pipeline_categorico, variables_categoricas_sel),
    ('categorico_cod', pipeline_categ_cod, variables_categ_cod_sel)
], verbose=True)

# Aplicar el preprocesamiento a los conjuntos de entrenamiento y prueba
X_entrenamiento_proc_1 = preprocesador_1.fit_transform(X_entrenamiento)
X_prueba_proc_1 = preprocesador_1.transform(X_prueba)


# In[27]:


# Obtener los nombres de las columnas numéricas
columnas_numericas_finales = variables_numericas_sel

# Obtener los nombres de las columnas generadas por el OneHotEncoder
columnas_categoricas_finales = preprocesador_1.named_transformers_['categorico']['codificador']\
    .get_feature_names_out(variables_categoricas_sel)

# Obtener las columnas categóricas codificadas (que no se transformaron con OneHotEncoder)
columnas_codificadas_finales = variables_categ_cod_sel

# Combinar todas las columnas procesadas en una sola lista
todas_las_columnas_1 = list(columnas_numericas_finales) + \
                       list(columnas_categoricas_finales) + \
                       list(columnas_codificadas_finales)

# Crear DataFrames a partir de los datos procesados
X_entrenamiento_df = pd.DataFrame(X_entrenamiento_proc_1, columns=todas_las_columnas_1)
X_prueba_df = pd.DataFrame(X_prueba_proc_1, columns=todas_las_columnas_1)

# Mostrar dimensiones y vista previa
print('Dimensiones del conjunto de entrenamiento:', X_entrenamiento_df.shape)
X_entrenamiento_df.head()


# In[28]:


# Pipeline numérico con PCA incluido: imputación, escalado y reducción de dimensionalidad
pipeline_numerico_pca = Pipeline([
    ('imputador', IterativeImputer(random_state=0, min_value=0)),
    ('escalador', MinMaxScaler()),
    ('pca', PCA(n_components=2, random_state=0))
])

# Crear un preprocesador que combine los tres tipos de variables
preprocesador_2 = ColumnTransformer([
    ('numerico', pipeline_numerico_pca, variables_numericas_sel),
    ('categorico', pipeline_categorico, variables_categoricas_sel),
    ('categorico_cod', pipeline_categ_cod, variables_categ_cod_sel)
], verbose=True)

# Aplicar el preprocesamiento con PCA a los conjuntos de entrenamiento y prueba
X_entrenamiento_proc_2 = preprocesador_2.fit_transform(X_entrenamiento)
X_prueba_proc_2 = preprocesador_2.transform(X_prueba)


# In[29]:


# Asignar nombres a los dos componentes principales del PCA
columnas_pca = [f'Componente_PCA_{i+1}' for i in range(2)]

# Obtener los nombres de las columnas creadas por el OneHotEncoder
columnas_categoricas_finales = preprocesador_2.named_transformers_['categorico']['codificador']\
    .get_feature_names_out(variables_categoricas_sel)

# Obtener las columnas categóricas codificadas (sin transformación)
columnas_codificadas_finales = variables_categ_cod_sel

# Combinar todas las columnas resultantes
todas_las_columnas_2 = list(columnas_pca) + \
                       list(columnas_categoricas_finales) + \
                       list(columnas_codificadas_finales)

# Crear DataFrames con los datos procesados (incluyendo PCA)
X_entrenamiento_df = pd.DataFrame(X_entrenamiento_proc_2, columns=todas_las_columnas_2)
X_prueba_df = pd.DataFrame(X_prueba_proc_2, columns=todas_las_columnas_2)

# Mostrar forma y primeras filas
print('Dimensiones del conjunto de entrenamiento:', X_entrenamiento_df.shape)
X_entrenamiento_df.head()


# Next, we will apply resampling techniques to handle data imbalance. Besides resampling, we can also adjust the decision threshold to further improve the model’s performance.
# 
# In the following code example, we will use SMOTENC (Synthetic Minority Over-sampling Technique for Nominal and Continuous features), which is specifically designed for datasets that contain both categorical and continuous variables. This method generates synthetic samples for the minority class while preserving the relationships between categorical and numerical features.

# In[30]:


# Identificar los índices de las variables categóricas en el conjunto procesado (sin PCA)
indices_categoricos_1 = list(
    range(len(variables_numericas_sel), X_entrenamiento_proc_1.shape[1])
)

# Aplicar sobremuestreo SMOTENC para balancear la clase minoritaria
smote_nc = SMOTENC(
    categorical_features=indices_categoricos_1,
    sampling_strategy='minority',
    random_state=42
)

# Generar el nuevo conjunto de entrenamiento balanceado (sin PCA)
X_entrenamiento_res_1, y_entrenamiento_res_1 = smote_nc.fit_resample(
    X_entrenamiento_proc_1, y_entrenamiento
)


# In[31]:


# Definir los índices de las variables categóricas en el conjunto procesado con PCA
# (Las dos primeras columnas corresponden a los componentes principales)
indices_categoricos_2 = list(range(2, X_entrenamiento_proc_2.shape[1]))

# Aplicar sobremuestreo SMOTENC para balancear la clase minoritaria
smote_nc = SMOTENC(
    categorical_features=indices_categoricos_2,
    sampling_strategy='minority',
    random_state=42
)

# Ajustar y generar el nuevo conjunto de entrenamiento balanceado (con PCA)
X_entrenamiento_res_2, y_entrenamiento_res_2 = smote_nc.fit_resample(
    X_entrenamiento_proc_2, y_entrenamiento
)


# In[32]:


# Resumen del balanceo de clases (conjunto sin PCA)
print('Ingeniería de características sin PCA\n')

print("Distribución original de clases:")
for clase in set(y_entrenamiento):
    cantidad = list(y_entrenamiento).count(clase)
    print(f"Clase {clase}: {cantidad} registros")

print("\nDistribución de clases después del sobremuestreo (SMOTENC):")
for clase in set(y_entrenamiento_res_1):
    cantidad = list(y_entrenamiento_res_1).count(clase)
    print(f"Clase {clase}: {cantidad} registros")


# In[33]:


# Resumen del balanceo de clases (conjunto con PCA)
print('Ingeniería de características con PCA\n')

print("Distribución original de clases:")
for clase in set(y_entrenamiento):
    cantidad = list(y_entrenamiento).count(clase)
    print(f"Clase {clase}: {cantidad} registros")

print("\nDistribución de clases después del sobremuestreo (SMOTENC):")
for clase in set(y_entrenamiento_res_2):
    cantidad = list(y_entrenamiento_res_2).count(clase)
    print(f"Clase {clase}: {cantidad} registros")


# In[34]:


# Crear un DataFrame con los datos balanceados (sin PCA) y sus nombres de columnas
X_resampleado_1 = pd.DataFrame(X_entrenamiento_res_1, columns=todas_las_columnas_1)

# Mostrar las primeras filas del conjunto re-muestreado
X_resampleado_1.head()


# In[35]:


# Crear un DataFrame con los datos balanceados (con PCA) y sus nombres de columnas
X_resampleado_2 = pd.DataFrame(X_entrenamiento_res_2, columns=todas_las_columnas_2)

# Mostrar las primeras filas del conjunto re-muestreado
X_resampleado_2.head()


# =============================
# GUARDAR DATASETS PROCESADOS
# =============================
import os, shutil
import pandas as pd

# 1) Crear carpetas destino
os.makedirs("data/processed/no_pca", exist_ok=True)
os.makedirs("data/processed/pca", exist_ok=True)
os.makedirs("data/processed/no_pca_smote", exist_ok=True)
os.makedirs("data/processed/pca_smote", exist_ok=True)

# 2) Sin PCA y sin balanceo
pd.DataFrame(X_entrenamiento_proc_1, columns=todas_las_columnas_1)\
  .to_csv("data/processed/no_pca/X_train.csv", index=False)
pd.DataFrame(X_prueba_proc_1, columns=todas_las_columnas_1)\
  .to_csv("data/processed/no_pca/X_test.csv", index=False)
pd.Series(y_entrenamiento, name="Churn").to_csv("data/processed/no_pca/y_train.csv", index=False)
pd.Series(y_prueba, name="Churn").to_csv("data/processed/no_pca/y_test.csv", index=False)

# 3) Con PCA y sin balanceo
pd.DataFrame(X_entrenamiento_proc_2, columns=todas_las_columnas_2)\
  .to_csv("data/processed/pca/X_train.csv", index=False)
pd.DataFrame(X_prueba_proc_2, columns=todas_las_columnas_2)\
  .to_csv("data/processed/pca/X_test.csv", index=False)
pd.Series(y_entrenamiento, name="Churn").to_csv("data/processed/pca/y_train.csv", index=False)
pd.Series(y_prueba, name="Churn").to_csv("data/processed/pca/y_test.csv", index=False)

# 4) Sin PCA + SMOTENC (SOLO train balanceado)
pd.DataFrame(X_entrenamiento_res_1, columns=todas_las_columnas_1)\
  .to_csv("data/processed/no_pca_smote/X_train.csv", index=False)
pd.Series(y_entrenamiento_res_1, name="Churn").to_csv("data/processed/no_pca_smote/y_train.csv", index=False)

# 5) Con PCA + SMOTENC (SOLO train balanceado)
pd.DataFrame(X_entrenamiento_res_2, columns=todas_las_columnas_2)\
  .to_csv("data/processed/pca_smote/X_train.csv", index=False)
pd.Series(y_entrenamiento_res_2, name="Churn").to_csv("data/processed/pca_smote/y_train.csv", index=False)

# 6) Copiar también los TEST a las carpetas SMOTE (mismos test que sus pares sin SMOTE)
shutil.copy("data/processed/no_pca/X_test.csv", "data/processed/no_pca_smote/X_test.csv")
shutil.copy("data/processed/no_pca/y_test.csv", "data/processed/no_pca_smote/y_test.csv")

shutil.copy("data/processed/pca/X_test.csv", "data/processed/pca_smote/X_test.csv")
shutil.copy("data/processed/pca/y_test.csv", "data/processed/pca_smote/y_test.csv")

print(" Guardados:")
print(" - data/processed/no_pca/{X_train,y_train,X_test,y_test}.csv")
print(" - data/processed/pca/{X_train,y_train,X_test,y_test}.csv")
print(" - data/processed/no_pca_smote/{X_train,y_train,X_test,y_test}.csv")
print(" - data/processed/pca_smote/{X_train,y_train,X_test,y_test}.csv")
