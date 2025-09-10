# Algoritmos Genéticos en Aprendizaje de Máquina

##  Objetivo
La presente actividad tiene como finalidad **comprender y aplicar los Algoritmos Genéticos (AG)** dentro del contexto del **aprendizaje de máquina (ML)**.  
Para ello se implementan tres casos prácticos:

1. **Feature Selection** → Uso de AG para seleccionar las características más relevantes de un dataset.  
2. **Hyperparameter Optimization** → Uso de AG para encontrar los mejores hiperparámetros de un modelo de ML.  
3. **Neuroevolution** → Uso de AG para evolucionar la arquitectura de una red neuronal.  

---

## Contexto
Los **Algoritmos Genéticos** son técnicas de optimización inspiradas en la evolución biológica.  
Su ciclo básico incluye:  

1. **Representación de la población (cromosomas):** posibles soluciones.  
2. **Inicialización:** creación aleatoria de la población inicial.  
3. **Función de aptitud (fitness):** mide qué tan buena es cada solución.  
4. **Selección:** elegir los mejores individuos para reproducirse.  
5. **Cruzamiento (crossover):** combinar individuos para crear descendencia.  
6. **Mutación:** introducir cambios aleatorios para mantener diversidad.  
7. **Terminación:** se detiene al alcanzar un número de generaciones o una aptitud óptima.  

---

##  Estructura del Repositorio

```bash
├── algoritmo de seleccion/
    └── main.py
    └── README.md
├── Hyperparameter Optimization/
    └── AGparaHyperparameterOptimization.ipynb
    └── README.md
├── neuroevolution/
    └── Neuroevolution.py
    └── README.md
├── algoritmos genéticos.pdf
└── README.md
