import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import random

class GeneticFeatureSelection:
    def __init__(self, n_features, population_size=50, n_generations=100, 
                 crossover_prob=0.8, mutation_prob=0.02, tournament_size=3,
                 alpha=0.8, beta=0.2):
        """
        Algoritmo Genético para Feature Selection
        
        Parámetros:
        - n_features: número total de características
        - population_size: tamaño de la población
        - n_generations: número de generaciones
        - crossover_prob: probabilidad de cruzamiento
        - mutation_prob: probabilidad de mutación
        - tournament_size: tamaño del torneo para selección
        - alpha: peso para la precisión del modelo
        - beta: peso para la simplicidad (menos características)
        """
        self.n_features = n_features
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.tournament_size = tournament_size
        self.alpha = alpha
        self.beta = beta
        
        # Para tracking
        self.historial_mejor_fitness = []
        self.historial_promedio_fitness = []
        self.mejor_individuo = None
        self.mejor_fitness = -np.inf
        
    def inicializar_poblacion(self):
        """Inicializa la población con cromosomas binarios aleatorios"""
        poblacion = []
        for _ in range(self.population_size):
            # Generar cromosoma con probabilidad 0.3-0.5 de tener gen=1
            cromosoma = np.random.choice([0, 1], size=self.n_features, p=[0.6, 0.4])
            # Asegurar que al menos una característica esté seleccionada
            if np.sum(cromosoma) == 0:
                cromosoma[np.random.randint(0, self.n_features)] = 1
            poblacion.append(cromosoma)
        return np.array(poblacion)
    
    def funcion_fitness(self, cromosoma, X, y):
        """
        Evalúa la aptitud de un cromosoma
        
        Fitness = α × Precisión_modelo + β × (1 - Ratio_características_seleccionadas)
        """
        # Obtener características seleccionadas
        caracteristicas_seleccionadas = np.where(cromosoma == 1)[0]
        
        if len(caracteristicas_seleccionadas) == 0:
            return 0  # Sin características seleccionadas
        
        # Entrenar modelo solo con características seleccionadas
        X_seleccionado = X[:, caracteristicas_seleccionadas]
        
        try:
            # Usar validación cruzada para obtener precisión
            modelo = SVC(kernel='linear', random_state=42)
            puntajes_cv = cross_val_score(modelo, X_seleccionado, y, cv=5, scoring='accuracy')
            precision = np.mean(puntajes_cv)
        except:
            precision = 0
        
        # Ratio de características seleccionadas
        ratio_caracteristicas = len(caracteristicas_seleccionadas) / self.n_features
        
        # Calcular fitness
        fitness = self.alpha * precision + self.beta * (1 - ratio_caracteristicas)
        
        return fitness
    
    def seleccion_torneo(self, poblacion, puntajes_fitness):
        """Selección por torneo"""
        seleccionados = []
        for _ in range(self.population_size):
            # Seleccionar participantes del torneo
            indices_torneo = np.random.choice(
                len(poblacion), size=self.tournament_size, replace=False
            )
            fitness_torneo = puntajes_fitness[indices_torneo]
            
            # Seleccionar el mejor del torneo
            indice_ganador = indices_torneo[np.argmax(fitness_torneo)]
            seleccionados.append(poblacion[indice_ganador].copy())
        
        return np.array(seleccionados)
    
    def cruzamiento(self, padre1, padre2):
        """Cruzamiento de un punto"""
        if np.random.random() > self.crossover_prob:
            return padre1.copy(), padre2.copy()
        
        # Seleccionar punto de cruzamiento
        punto_cruzamiento = np.random.randint(1, self.n_features)
        
        # Crear hijos
        hijo1 = np.concatenate([padre1[:punto_cruzamiento], padre2[punto_cruzamiento:]])
        hijo2 = np.concatenate([padre2[:punto_cruzamiento], padre1[punto_cruzamiento:]])
        
        return hijo1, hijo2
    
    def mutacion(self, cromosoma):
        """Mutación bit-flip"""
        mutado = cromosoma.copy()
        for i in range(len(mutado)):
            if np.random.random() < self.mutation_prob:
                mutado[i] = 1 - mutado[i]  # Flip bit
        
        # Asegurar que al menos una característica esté seleccionada
        if np.sum(mutado) == 0:
            mutado[np.random.randint(0, self.n_features)] = 1
            
        return mutado
    
    def evolucionar(self, X, y):
        """Ejecuta el algoritmo genético"""
        # Inicializar población
        poblacion = self.inicializar_poblacion()
        
        print("🧬 Iniciando evolución...")
        print(f"📊 Población: {self.population_size}, Generaciones: {self.n_generations}")
        print(f"🎯 Características totales: {self.n_features}")
        print("-" * 60)
        
        for generacion in range(self.n_generations):
            # Evaluar fitness de toda la población
            puntajes_fitness = np.array([
                self.funcion_fitness(individuo, X, y) for individuo in poblacion
            ])
            
            # Tracking del mejor individuo
            mejor_indice = np.argmax(puntajes_fitness)
            if puntajes_fitness[mejor_indice] > self.mejor_fitness:
                self.mejor_fitness = puntajes_fitness[mejor_indice]
                self.mejor_individuo = poblacion[mejor_indice].copy()
            
            # Guardar estadísticas
            self.historial_mejor_fitness.append(np.max(puntajes_fitness))
            self.historial_promedio_fitness.append(np.mean(puntajes_fitness))
            
            # Imprimir progreso cada 25 generaciones
            if generacion % 25 == 0 or generacion == self.n_generations - 1:
                n_seleccionadas = np.sum(self.mejor_individuo)
                reduccion = (1 - n_seleccionadas/self.n_features) * 100
                print(f"Gen {generacion:3d}: Mejor={self.mejor_fitness:.3f} | "
                      f"Promedio={np.mean(puntajes_fitness):.3f} | "
                      f"Características={n_seleccionadas}/{self.n_features} ({reduccion:.1f}% reducción)")
            
            # Selección
            poblacion_seleccionada = self.seleccion_torneo(poblacion, puntajes_fitness)
            
            # Cruzamiento y mutación
            nueva_poblacion = []
            for i in range(0, self.population_size, 2):
                padre1 = poblacion_seleccionada[i]
                padre2 = poblacion_seleccionada[(i + 1) % self.population_size]
                
                hijo1, hijo2 = self.cruzamiento(padre1, padre2)
                hijo1 = self.mutacion(hijo1)
                hijo2 = self.mutacion(hijo2)
                
                nueva_poblacion.extend([hijo1, hijo2])
            
            poblacion = np.array(nueva_poblacion[:self.population_size])
        
        print("-" * 60)
        print("✅ Evolución completada!")
    
    def obtener_resultados(self, nombres_caracteristicas=None):
        """Obtiene los resultados del algoritmo genético"""
        caracteristicas_seleccionadas = np.where(self.mejor_individuo == 1)[0]
        
        resultados = {
            'cromosoma': self.mejor_individuo,
            'indices_caracteristicas_seleccionadas': caracteristicas_seleccionadas,
            'n_caracteristicas_seleccionadas': len(caracteristicas_seleccionadas),
            'reduccion_caracteristicas': (1 - len(caracteristicas_seleccionadas)/self.n_features) * 100,
            'mejor_fitness': self.mejor_fitness
        }
        
        if nombres_caracteristicas is not None:
            resultados['nombres_caracteristicas_seleccionadas'] = [nombres_caracteristicas[i] for i in caracteristicas_seleccionadas]
        
        return resultados
    
    def graficar_evolucion(self):
        """Grafica la evolución del fitness"""
        plt.figure(figsize=(12, 5))
        
        # Gráfico de evolución del fitness
        plt.subplot(1, 2, 1)
        generaciones = range(len(self.historial_mejor_fitness))
        plt.plot(generaciones, self.historial_mejor_fitness, 'b-', label='Mejor Fitness', linewidth=2)
        plt.plot(generaciones, self.historial_promedio_fitness, 'r--', label='Fitness Promedio', alpha=0.7)
        plt.xlabel('Generación')
        plt.ylabel('Fitness')
        plt.title('Evolución del Fitness')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Gráfico del mejor cromosoma
        plt.subplot(1, 2, 2)
        plt.bar(range(len(self.mejor_individuo)), self.mejor_individuo, 
                color=['green' if gen == 1 else 'red' for gen in self.mejor_individuo])
        plt.xlabel('Característica')
        plt.ylabel('Seleccionada (1) / No Seleccionada (0)')
        plt.title('Mejor Cromosoma Encontrado')
        plt.xticks(range(len(self.mejor_individuo)))
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def principal():
    """Función principal para demostrar el algoritmo con el dataset Iris"""
    print("🌸 Cargando dataset Iris...")
    
    # Cargar dataset Iris
    dataset = load_iris()
    X = dataset.data
    y = dataset.target
    nombres_caracteristicas = dataset.feature_names
    
    # Estandarizar datos
    escalador = StandardScaler()
    X_escalado = escalador.fit_transform(X)
    
    print(f"📊 Dataset: {X.shape[0]} muestras, {X.shape[1]} características")
    print(f"🏷️  Características: {nombres_caracteristicas}")
    print()
    
    # Crear y ejecutar algoritmo genético
    algoritmo_genetico = GeneticFeatureSelection(
        n_features=X.shape[1],
        population_size=50,
        n_generations=100,
        crossover_prob=0.8,
        mutation_prob=0.02,
        tournament_size=3,
        alpha=0.8,  # Peso para precisión
        beta=0.2    # Peso para simplicidad
    )
    
    # Ejecutar evolución
    algoritmo_genetico.evolucionar(X_escalado, y)
    
    # Obtener y mostrar resultados
    resultados = algoritmo_genetico.obtener_resultados(nombres_caracteristicas)
    
    print()
    print("🎯 RESULTADOS FINALES:")
    print("-" * 40)
    print(f"Cromosoma final: {resultados['cromosoma']}")
    print(f"Características seleccionadas: {resultados['n_caracteristicas_seleccionadas']}/{len(nombres_caracteristicas)}")
    print(f"Reducción de características: {resultados['reduccion_caracteristicas']:.1f}%")
    print(f"Mejor fitness: {resultados['mejor_fitness']:.4f}")
    print()
    print("🌟 Características seleccionadas:")
    for i, nombre in enumerate(resultados['nombres_caracteristicas_seleccionadas']):
        print(f"  • {nombre}")
    
    # Validar resultado final
    print()
    print("🔍 VALIDACIÓN FINAL:")
    print("-" * 30)
    
    # Modelo con todas las características
    modelo_todas = SVC(kernel='linear', random_state=42)
    puntajes_todas = cross_val_score(modelo_todas, X_escalado, y, cv=5, scoring='accuracy')
    print(f"Accuracy con TODAS las características: {np.mean(puntajes_todas):.4f} ± {np.std(puntajes_todas):.4f}")
    
    # Modelo con características seleccionadas
    indices_seleccionados = resultados['indices_caracteristicas_seleccionadas']
    X_seleccionado = X_escalado[:, indices_seleccionados]
    modelo_seleccionado = SVC(kernel='linear', random_state=42)
    puntajes_seleccionados = cross_val_score(modelo_seleccionado, X_seleccionado, y, cv=5, scoring='accuracy')
    print(f"Accuracy con características SELECCIONADAS: {np.mean(puntajes_seleccionados):.4f} ± {np.std(puntajes_seleccionados):.4f}")
    
    mejora = np.mean(puntajes_seleccionados) - np.mean(puntajes_todas)
    print(f"Mejora en accuracy: {mejora:+.4f}")
    
    # Graficar evolución
    algoritmo_genetico.graficar_evolucion()


if __name__ == "__main__":
    principal()