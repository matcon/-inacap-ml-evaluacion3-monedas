# Evaluación 3: Machine Learning - Clasificador CNN de Monedas

Proyecto de Deep Learning para la asignatura de Machine Learning (INACAP). Consiste en el diseño, entrenamiento y evaluación de una Red Neuronal Convolucional (CNN) para la clasificación binaria de monedas de $100 pesos chilenos (**Cara** vs. **Sello**), incluyendo el análisis ante objetos fuera de distribución.

---

## 📁 Estructura del Repositorio

```text
├── E3_Contreras.ipynb    # Notebook principal con todo el flujo de ML y análisis
├── dataset/              # Dataset de entrenamiento y validación (100 imágenes)
│   ├── cara/             # 50 imágenes de la clase CARA (etiqueta 0)
│   └── sello/            # 50 imágenes de la clase SELLO (etiqueta 1)
└── pruebas/              # Imágenes independientes para pruebas de inferencia real
    ├── prueba_cara_1.jpg
    ├── prueba_sello_1.jpg
    └── prueba_no_entrenada_pila_cr2032.jpg
```

---

## 🪙 Dataset y Preprocesamiento

* **Origen de datos:** Extraído a partir de grabaciones de video en alta definición (4K) de monedas de $100 pesos chilenos bajo diversos ángulos y condiciones de iluminación.
* **Auto-Recorte Centrado (*Auto-Crop*):** Mediante la **Transformada Circular de Hough** en OpenCV, se detecta el círculo exacto de la moneda y se realiza un recorte cuadrado con 20% de margen de seguridad, redimensionado a $256 \times 256$ píxeles con interpolación Lanczos4. Esto maximiza el área útil de la moneda al 85% del encuadre y elimina el fondo innecesario.
* **Filtro de Nitidez Temporal:** Se evaluó la varianza del Laplaciano en bloques temporales de los videos para descartar fotogramas con desenfoque de movimiento (*motion blur*), asegurando máxima nitidez en el relieve de las monedas.

---

## 🧠 Arquitectura y Experimentos

El notebook evalúa dos arquitecturas secuenciales en TensorFlow/Keras con resolución de entrada de $64 \times 64 \times 3$:

| Parámetro / Componente | Experimento 1 (Base) | Experimento 2 (Mejorado) |
| :--- | :--- | :--- |
| **Capas Convolucionales** | 2 bloques (32, 64 filtros de $3\times3$) | 3 bloques (32, 64, 64 filtros de $3\times3$) |
| **Max Pooling** | Sí ($2\times2$) tras cada Conv2D | Sí ($2\times2$) tras cada Conv2D |
| **Regularización** | Ninguna | `Dropout(0.2)` en capa densa |
| **Capas Densas** | Dense(64, ReLU) $\rightarrow$ Dense(1, Sigmoid) | Dense(64, ReLU) $\rightarrow$ Dense(1, Sigmoid) |
| **Optimizador y Loss** | Adam, Binary Crossentropy | Adam, Binary Crossentropy |
| **Épocas** | 15 épocas | 20 épocas |
| **Accuracy Validación** | **70.0%** | **80.0%** |

---

## 🔍 Resultados de Inferencia en Pruebas Reales

Se evaluó el modelo final guardado (`modelo_monedas_cnn.keras`) con imágenes de prueba independientes:

1. **Prueba Real Cara (`prueba_cara_1.jpg`):**
   * Predicción: **CARA** (Probabilidad: 75.3% / Sigmoide: 0.2469) $\rightarrow$ **Acierto correcto**.
2. **Prueba Real Sello (`prueba_sello_1.jpg`):**
   * Predicción: **SELLO** (Probabilidad: 90.8% / Sigmoide: 0.9077) $\rightarrow$ **Acierto correcto**.
3. **Categoría No Entrenada (`prueba_no_entrenada_pila_cr2032.jpg`):**
   * Objeto: Pila botón metálica circular (CR2032).
   * Predicción: **CARA** (Sigmoide: 0.0957).
   * **Análisis de Limitaciones:** La neurona de salida única con activación `sigmoid` confina matemáticamente el valor entre 0 y 1, forzando la clasificación a una de las dos clases entrenadas al no existir una neurona para "objeto desconocido". La reflectividad metálica y forma circular activan los filtros convolucionales a pesar de no ser una moneda.

---

## 🚀 Requisitos e Instalación

Para ejecutar el notebook localmente:

```bash
# Clonar el repositorio
git clone git@github.com:matcon/-inacap-ml-evaluacion3-monedas.git
cd -inacap-ml-evaluacion3-monedas

# Crear y activar entorno virtual
python3 -m venv .venv
source .venv/bin/activate

# Instalar dependencias
pip install tensorflow opencv-python matplotlib numpy
```

Abrir y ejecutar las celdas en Jupyter Notebook o VS Code:
```bash
jupyter notebook E3_Contreras.ipynb
```
