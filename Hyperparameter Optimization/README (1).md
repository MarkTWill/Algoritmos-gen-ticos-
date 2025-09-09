# 🧬 Algoritmos Genéticos en Machine Learning

Este repositorio contiene ejemplos del uso de **algoritmos genéticos (AG)** en diferentes tareas de *machine learning*:  

1. **Feature Selection** → Selección de características más relevantes.  
2. **Hyperparameter Optimization** → Optimización automática de hiperparámetros.  
3. **Neuroevolution** → Búsqueda de arquitecturas de redes neuronales.  

---

## 🎯 Objetivo
Demostrar cómo los **algoritmos genéticos**, inspirados en la evolución biológica, pueden aplicarse para **optimizar modelos de aprendizaje de máquina**, mejorando su rendimiento y reduciendo el trabajo manual.

---

## 🔄 Flujo de un Algoritmo Genético
1. **Población inicial** → soluciones generadas al azar (ej. combinaciones de hiperparámetros).  
2. **Fitness** → medir qué tan buena es cada solución (ej. accuracy del modelo).  
3. **Selección** → elegir los mejores individuos.  
4. **Cruzamiento (crossover)** → combinar soluciones para crear nuevas.  
5. **Mutación** → cambiar aleatoriamente algún valor para explorar más opciones.  
6. **Iteración** → repetir varias generaciones hasta encontrar la mejor solución.  

---

## 📌 Ejemplo 1: Hyperparameter Optimization (SVM en Iris)

En este ejemplo, se usa un AG para optimizar los hiperparámetros de un **SVM** (`C`, `gamma`, `kernel`) sobre el dataset **Iris**.

### 🔧 Requisitos
```bash
pip install scikit-learn num