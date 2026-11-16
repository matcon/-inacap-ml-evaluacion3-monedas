# Evaluación 3: Machine Learning - Deep Learning
## Clasificación de Monedas de $100 Centenario (Cara vs Sello) mediante Redes Neuronales Convolucionales (CNN)

* **Institución:** INACAP - Sede Renca / Santiago
* **Asignatura:** Machine Learning (TIHI84) - 8vo Semestre
* **Estudiante:** Pedro Matías Contreras Lagos
* **Fecha:** Noviembre 2026

---

## Descripción del Proyecto

Este repositorio contiene la implementación y entrega de la **Evaluación 3 de Machine Learning**, correspondiente al diseño, entrenamiento y evaluación de un clasificador de imágenes mediante **Redes Neuronales Convolucionales (CNN)** en TensorFlow / Keras.

El objetivo consiste en clasificar imágenes de la nueva moneda bimetálica conmemorativa de $100 emitida por el Banco Central de Chile (1925–2025) en dos clases observables:
1. **`cara` (Anverso Conmemorativo - Etiqueta 0):** Presenta el isotipo del centenario, la inscripción central **"CIEN años"**, la Cordillera de los Andes y la leyenda *"REPÚBLICA DE CHILE - BANCO CENTRAL DE CHILE 1925-2025"*.
2. **`sello` (Reverso Oficial - Etiqueta 1):** Presenta el **Escudo Nacional de Chile**, el valor facial **"100 PESOS"**, el año de acuñación *"2025"* y la orla de laureles.

---

## Estructura del Repositorio

```text
.
├── dataset/
│   ├── cara/                               # 50 imágenes originales (256x256 RGB, .jpg)
│   └── sello/                              # 50 imágenes originales (256x256 RGB, .jpg)
├── pruebas/
│   ├── prueba_cara_1.jpg                   # Muestra independiente de validación Anverso
│   ├── prueba_cara_2.jpg                   # Muestra independiente de validación Anverso
│   ├── prueba_sello_1.jpg                  # Muestra independiente de validación Reverso
│   ├── prueba_sello_2.jpg                  # Muestra independiente de validación Reverso
│   ├── prueba_sello_3.jpg                  # Muestra independiente de validación Reverso
│   ├── prueba_no_entrenada_pila_cr2032.jpg  # Categoría NO entrenada (Pila botón CR2032 / OOD)
│   ├── prueba_no_entrenada_golilla.jpg      # Categoría NO entrenada adicional (Arandela / OOD)
│   └── originales/                         # Fotografías originales sin procesar
├── E3_Contreras.ipynb                       # Cuaderno Jupyter oficial con la entrega completa
├── modelo_monedas_cnn.keras                 # Modelo final entrenado y serializado en Keras 3
├── capturar_desde_video.py                  # Script para extracción de frames desde video
├── README.md                               # Documentación principal del repositorio
└── Evaluacion_3_Machine_Learning.pdf        # Pauta y rúbrica oficial de la evaluación
```

---

## Contenido del Cuaderno (`E3_Contreras.ipynb`)

El notebook se estructura según los 10 indicadores de la rúbrica oficial:
1. **Identificación y Formulación del Problema:** Definición de objetivos, variables $X, y$, caso de uso y justificación de por qué CNN frente a MLP.
2. **Conjunto de Datos:** Registro de 50 imágenes por clase (100 totales), origen propio, permisos y visualización.
3. **Preparación y Aumento de Datos:** Partición 80% train / 20% val (`seed=42`), normalización y Data Augmentation con ejemplo "Antes y Después".
4. **Diseño de la Red Neuronal:** Definición capa por capa de la CNN (`Conv2D`, `MaxPooling2D`, `Flatten`, `Dense`, `Dropout`, `sigmoid`).
5. **Entrenamiento y Ajuste Experimental:** Ejecución de 2 experimentos comparativos y gráficos de curvas de rendimiento (Accuracy y Loss).
6. **Modelo Guardado, Inferencia y Limitaciones:** Guardado y recarga en `.keras`, función `interpretar(valor)` y evaluación de muestras reales y objeto no entrenado (pila CR2032).
7. **Conclusiones:** Logro de objetivos, aprendizajes clave y propuestas de mejora.
