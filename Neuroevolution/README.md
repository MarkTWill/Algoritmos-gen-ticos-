
#  Neuroevolution con Algoritmo Genético (Caso Iris)

Este proyecto muestra un ejemplo práctico de **Neuroevolution**: el uso de **algoritmos genéticos (AG)** para diseñar y optimizar automáticamente la arquitectura de redes neuronales.  
El caso de aplicación es el **dataset Iris**, un clásico en Machine Learning para la clasificación de flores .

---

##  Objetivo
El objetivo es **automatizar la búsqueda de la mejor arquitectura de red neuronal** (número de neuronas en capas ocultas y tasa de aprendizaje) mediante un algoritmo genético.

---

## Funcionamiento del Algoritmo Genético

1. **Representación de cromosomas**  
   Cada individuo de la población representa una red neuronal definida por:  
   - `neurons1`: número de neuronas en la primera capa oculta.  
   - `neurons2`: número de neuronas en la segunda capa oculta.  
   - `lr`: tasa de aprendizaje.  

2. **Inicialización**  
   Se crea una población inicial de arquitecturas aleatorias.  

3. **Función de aptitud (fitness)**  
   Cada individuo se evalúa entrenando su red neuronal en el dataset Iris y midiendo la **precisión en validación**.  

4. **Selección**  
   Se aplica **torneo** entre varios individuos y se escoge el mejor para reproducirse.  

5. **Cruce (crossover)**  
   Se combinan parámetros de dos padres para generar un hijo.  

6. **Mutación**  
   Con cierta probabilidad se alteran neuronas o tasa de aprendizaje, introduciendo diversidad.  

7. **Criterio de terminación**  
   El proceso se repite varias generaciones. El mejor individuo final es la arquitectura más apta.  

---

##  Caso de estudio: Dataset Iris

- **Características (X):** Largo y ancho de sépalos y pétalos.  
- **Clases (y):** 3 tipos de flores (*Setosa, Versicolor, Virginica*).  
- Se usa `StandardScaler` para normalizar las variables.  
- División: 80% entrenamiento, 20% prueba.  

---


## 🚀 Ejecución

### 1. Clonar el repositorio
```bash
git clone https://github.com/tuusuario/neuroevolution-iris.git
cd neuroevolution-iris
