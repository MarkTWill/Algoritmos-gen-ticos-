import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
iris = load_iris()
X = iris.data
y = to_categorical(iris.target)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =====================
def crear_individuo():
    return {
        "neurons1": np.random.randint(4, 32),
        "neurons2": np.random.randint(4, 32),
        "lr": 10**np.random.uniform(-3, -1)
    }

def evaluar(ind):
    model = Sequential([
        Dense(ind["neurons1"], activation="relu", input_shape=(4,)),
        Dense(ind["neurons2"], activation="relu"),
        Dense(3, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(X_train, y_train, epochs=10, batch_size=8, verbose=0, validation_data=(X_test, y_test))
    acc = history.history["val_accuracy"][-1]
    return acc

def seleccionar(poblacion, fitness):
    idx = np.random.choice(len(poblacion), 3, replace=False)
    idx = sorted(idx, key=lambda i: fitness[i], reverse=True)
    return poblacion[idx[0]]

def cruzar(p1, p2):
    return {
        "neurons1": np.random.choice([p1["neurons1"], p2["neurons1"]]),
        "neurons2": np.random.choice([p1["neurons2"], p2["neurons2"]]),
        "lr": (p1["lr"] + p2["lr"]) / 2
    }

def mutar(ind):
    if np.random.rand() < 0.3:
        ind["neurons1"] = np.random.randint(4, 32)
    if np.random.rand() < 0.3:
        ind["neurons2"] = np.random.randint(4, 32)
    if np.random.rand() < 0.3:
        ind["lr"] = 10**np.random.uniform(-3, -1)
    return ind

# =====================
# Proceso evolutivo
# =====================
def neuroevolution(num_generaciones=5, tam_poblacion=6):
    poblacion = [crear_individuo() for _ in range(tam_poblacion)]
    mejores, promedios, logs = [], [], []
    historico_arq = []

    for g in range(num_generaciones):
        fitness = [evaluar(ind) for ind in poblacion]
        mejor = max(fitness)
        promedio = np.mean(fitness)

        mejores.append(mejor)
        promedios.append(promedio)
        logs.append(f"Gen {g+1}: Mejor={mejor:.3f}, Promedio={promedio:.3f}")
        historico_arq.extend(poblacion)

        nueva_poblacion = []
        for _ in range(tam_poblacion):
            p1, p2 = seleccionar(poblacion, fitness), seleccionar(poblacion, fitness)
            hijo = cruzar(p1, p2)
            hijo = mutar(hijo)
            nueva_poblacion.append(hijo)
        poblacion = nueva_poblacion

    # Mejor individuo final
    fitness = [evaluar(ind) for ind in poblacion]
    idx_best = np.argmax(fitness)
    mejor_ind = poblacion[idx_best]
    mejor_acc = fitness[idx_best]
    # =============================================================
    # Evolución de precisión
    plt.figure()
    plt.plot(range(1, num_generaciones+1), mejores, label="Mejor")
    plt.plot(range(1, num_generaciones+1), promedios, label="Promedio")
    plt.xlabel("Generaciones")
    plt.ylabel("Precisión")
    plt.title("Evolución de Precisión")
    plt.legend()
    path1 = "/tmp/evolucion.png"
    plt.savefig(path1)
    plt.close()

    # Comparación final
    plt.figure()
    plt.bar(["Mejor", "Promedio"], [mejores[-1], promedios[-1]])
    plt.title("Comparación Final")
    path2 = "/tmp/comparacion.png"
    plt.savefig(path2)
    plt.close()

    # Histograma de neuronas
    plt.figure()
    plt.hist([h["neurons1"] for h in historico_arq], bins=10, alpha=0.5, label="Capa 1")
    plt.hist([h["neurons2"] for h in historico_arq], bins=10, alpha=0.5, label="Capa 2")
    plt.title("Distribución de Neuronas")
    plt.legend()
    path3 = "/tmp/neuronas.png"
    plt.savefig(path3)
    plt.close()

    # Histograma tasa aprendizaje
    plt.figure()
    plt.hist([h["lr"] for h in historico_arq], bins=10, color="orange")
    plt.title("Distribución Tasa de Aprendizaje")
    path4 = "/tmp/lr.png"
    plt.savefig(path4)
    plt.close()
    resumen = f"Mejor arquitectura encontrada:\nNeuronas capa1: {mejor_ind['neurons1']} | Neuronas capa2: {mejor_ind['neurons2']} | LR: {mejor_ind['lr']:.4f} | Acc: {mejor_acc:.3f}"
    return resumen, path1, path2, path3, path4, "\n".join(logs)

# =============================================================================

iface = gr.Interface(
    fn=neuroevolution,
    inputs=[
        gr.Slider(3, 10, value=5, step=1, label="Número de generaciones"),
        gr.Slider(4, 12, value=6, step=1, label="Tamaño de población")
    ],
    outputs=[
        gr.Textbox(label="Resumen"),
        gr.Image(type="filepath", label="Evolución de precisión"),
        gr.Image(type="filepath", label="Comparación final"),
        gr.Image(type="filepath", label="Histograma de neuronas"),
        gr.Image(type="filepath", label="Histograma LR"),
        gr.Textbox(label="Log evolutivo")
    ],
    title="Neuroevolution con Algoritmo Genético (Caso Iris)",
    description="Ejemplo práctico de AG aplicados a redes neuronales: optimización automática de arquitectura sobre el dataset Iris."
)

iface.launch(debug=True)