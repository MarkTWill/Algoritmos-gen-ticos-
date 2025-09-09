# Algoritmo Genético para Selección de Características

implementación del **Algoritmo Genético (GA)** para la
**selección de características** (feature selection) usando el dataset
**Iris**.\
El objetivo es encontrar un subconjunto óptimo de características que
maximice la precisión del modelo y minimice la cantidad de atributos
seleccionados.

## 📂 Contenido

-   `main.py`: Código principal del algoritmo.
-   Uso del dataset Iris de `scikit-learn`.
-   Clasificación mediante **SVM** con validación cruzada.

## ⚙️ Funcionamiento

El algoritmo utiliza: - **Codificación binaria** de cromosomas (1 =
característica seleccionada, 0 = descartada). - **Selección por
torneo**. - **Cruzamiento de un punto**. - **Mutación bit-flip**. -
**Función de fitness** basada en: - Precisión del modelo (SVM). -
Proporción de características seleccionadas.

## 🧬 Parámetros principales

-   `population_size`: Tamaño de la población.
-   `n_generations`: Número de generaciones.
-   `crossover_prob`: Probabilidad de cruzamiento.
-   `mutation_prob`: Probabilidad de mutación.
-   `alpha`: Peso de la precisión en el fitness.
-   `beta`: Peso de la simplicidad en el fitness.

## 📊 Resultados

El algoritmo muestra: - Evolución del fitness (mejor y promedio por
generación). - Cromosoma final con características seleccionadas. -
Comparación del accuracy entre usar todas las características vs. las
seleccionadas.

## 🚀 Ejecución

Instala las dependencias necesarias:

``` bash
pip install numpy pandas scikit-learn matplotlib
```

Ejecuta el archivo principal:

``` bash
python main.py
```

## 📈 Ejemplo de salida

-   Accuracy con todas las características.
-   Accuracy con las características seleccionadas.
-   Reducción en el número de características.
-   Gráficas de evolución del fitness.

## 📝 Notas

-   El dataset utilizado es **Iris** incluido en `scikit-learn`.
-   Puede adaptarse fácilmente a otros datasets reemplazando la carga de
    datos.
