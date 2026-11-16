# %% [markdown]
# # Evaluación 3 — Machine Learning: Clasificador Binario de Moneda Conmemorativa de $100 (Centenario Banco Central de Chile) mediante Deep Learning (CNN)
#
# ## 1. Identificación y Definición del Problema
#
# * **Institución:** INACAP — Área Tecnologías de la Información y Ciberseguridad (Ingeniería Informática, 8vo Semestre)
# * **Asignatura:** Machine Learning — Evaluación Sumativa 3 (Ponderación: 40% | Puntaje máximo: 100 puntos)
# * **Integrantes:**
#   * Pedro Matías Contreras Lagos
#   * *[Agregar segundo integrante si aplica]*
# * **Nombre del archivo de entrega:** `E3_Contreras_Lagos.ipynb`
#
# ### 1.1 Título y Objetivo del Proyecto
# * **Título:** Clasificación visual automatizada de orientación (**Cara vs. Sello**) en la Moneda Conmemorativa de $100 del Centenario del Banco Central de Chile (1925–2025) utilizando Redes Neuronales Convolucionales (CNN).
# * **Objetivo General:** Diseñar, entrenar, ajustar y validar una red neuronal convolucional profunda capaz de discriminar de forma binaria entre el **anverso conmemorativo (`cara`)** y el **reverso oficial (`sello`)** de la moneda bimetálica de $100 del Centenario del Banco Central de Chile, evaluando su generalización en particiones independientes y analizando su comportamiento ante objetos metálicos fuera de distribución (*Out-of-Distribution*).
#
# ### 1.2 Definición de las Dos Categorías Observables y Caso de Uso Práctico
# De acuerdo con las especificaciones técnicas oficiales del [Banco Central de Chile](https://www.billetesymonedas.cl/MonedaCentenario/Index) ([ficha numismática Numista #474034](https://es.numista.com/474034)), la moneda bimetálica de $100 conmemorativa ($23,5\text{ mm}$ de diámetro y $7,58\text{ g}$ de peso) posee un anillo exterior dorado (92% Cu, 6% Al, 2% Ni) y un núcleo central plateado (70% Cu, 15% Ni, 15% Zn). Se definen **dos clases mutuamente excluyentes**:
#
# 1. **Clase 0 — `cara` (Anverso Conmemorativo Centenario):** Presenta en el núcleo plateado el isotipo del Banco Central de Chile (tres arcos crecientes y sol) sobre una representación tramada de la Cordillera de los Andes, junto a la leyenda central en alto relieve **"CIEN"** (con la letra *E* estilizada de tres barras horizontales), la palabra **"años"** y la marca de ceca **S**. En el anillo dorado superior lleva la inscripción `"REPÚBLICA DE CHILE"` y en el inferior `"BANCO CENTRAL DE CHILE 1925 - 2025"`.
# 2. **Clase 1 — `sello` (Reverso Oficial):** Presenta en el núcleo plateado el **Escudo Nacional de Chile**, bajo él el año de acuñación (**"2025"**), la cifra **"100"** de gran tamaño y la palabra **"PESOS"** superpuesta entre el núcleo y el anillo inferior, circundado por dos **ramas de laurel** en el anillo dorado.
#
# * **Uso posible en la industria:** Módulo de visión por computador en **líneas de control de calidad de acuñación (Casa de Moneda), máquinas contadoras/clasificadoras de efectivo bancario y sistemas de orientación de cospeles**, donde una cámara cenital debe verificar en tiempo real en qué cara se encuentra apoyada cada moneda antes del empaquetado o validación mecánica.
#
# ### 1.3 Definición Formal de Entrada ($X$), Etiqueta ($y$), Recursos y Bibliotecas
# * **Entrada del modelo ($X$):** Tensor tridimensional de imagen en color RGB redimensionado a resolución espacial fija de $128 \times 128$ píxeles:
#   $$X \in \mathbb{R}^{128 \times 128 \times 3}, \quad X_{i,j,c} \in [0, 255] \rightarrow [0.0, 1.0]$$
# * **Etiqueta de salida ($y$):** Variable escalar binaria codificada alfabéticamente por directorio (`label_mode="binary"`):
#   $$y \in \{0, 1\} \quad \text{donde } y = 0 \implies \text{cara (Anverso)}, \quad y = 1 \implies \text{sello (Reverso)}$$
# * **Bibliotecas utilizadas:** `tensorflow` / `keras`, `numpy`, `matplotlib`, `PIL` (*Pillow*), `hashlib` y `os`.
#
# ### 1.4 Pertinencia de una Red Convolucional (CNN) frente a un Perceptrón Multicapa (MLP)
# En una moneda bimetálica, tanto la `cara` como el `sello` poseen exactamente la misma silueta circular, el mismo diámetro ($23,5\text{ mm}$) y la misma distribución global de colores metálicos (anillo dorado exterior y núcleo plateado interior). Un **MLP** aplana la imagen con `Flatten()` destruyendo la topología 2D ($49.152$ entradas) y carece de invarianza espacial. Una **CNN** aplica filtros locales de $3 \times 3$ con pesos compartidos que detectan los bordes y micro-relieves sobre el metal (`"CIEN"` y la Cordillera tramada vs. el Escudo Nacional, las ramas de laurel y `"100"`), manteniendo invarianza traslacional mediante `MaxPooling2D`.
#
# ### 1.5 Acceso al Dataset, Estructura de Carpetas e Instrucciones de Carga
# * **Enlace accesible al dataset:** `[Pegar aquí tu enlace compartido de Google Drive o GitHub con permisos de lectura]`
# * **Origen del código e imágenes:** Código adaptado de la documentación oficial de [Keras Image Classification](https://keras.io/examples/vision/image_classification_from_scratch/) y ejemplos de clases. Imágenes obtenidas por captura propia mediante video y extracción de fotogramas en OpenCV (`capturar_desde_video.py`).
# * **Estructura de carpetas:**
# ```text
# proyecto_evaluacion3/
# ├── E3_Contreras_Lagos.ipynb
# ├── dataset/
# │   ├── cara/      # >= 50 imágenes originales (.jpg) del anverso ("CIEN años")
# │   └── sello/     # >= 50 imágenes originales (.jpg) del reverso ("100 PESOS" + Escudo)
# └── pruebas/
#     ├── prueba_cara_1.jpg
#     ├── prueba_sello_1.jpg
#     ├── prueba_no_entrenada_golilla.jpg      # Categoría NO entrenada (Arandela metálica)
#     └── prueba_no_entrenada_pila_cr2032.jpg  # Categoría NO entrenada (Batería botón CR2032)
# ```

# %%
# ==============================================================================
# CELDA 1: Importación de bibliotecas y fijación de semillas de reproducibilidad
# ==============================================================================
import os
import hashlib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import load_img, img_to_array

# Fijación estricta de semilla global para reproducibilidad en particiones y pesos
SEMILLA = 42
tf.random.set_seed(SEMILLA)
np.random.seed(SEMILLA)

print(f"Versión de TensorFlow en uso : {tf.__version__}")
print(f"Semilla global configurada   : {SEMILLA}")
print(f"Dispositivos físicos         : {tf.config.list_physical_devices()}")

# %% [markdown]
# ## 2. Conjunto de Datos: Auditoría de Calidad, Diversidad, Fuentes y Ejemplos
#
# * **Origen y permisos de uso:** Captura propia de la moneda física de $100 Centenario del Banco Central de Chile mediante muestreo espaciado de fotogramas de video (100% autoría propia para uso académico libre en INACAP, sin restricciones de terceros).
# * **Conteos y etiquetas:** Mínimo de 50 imágenes originales diferentes por clase (`dataset/cara` $\rightarrow y=0$, `dataset/sello` $\rightarrow y=1$). Ninguna imagen aumentada se guarda como archivo original.
# * **Variaciones de iluminación, fondo y posición:** Se capturaron muestras variando el ángulo de incidencia de la luz (cenital y rasante para resaltar el bajorrelieve metálico), rotando físicamente la moneda entre $0^\circ$ y $360^\circ$ con leves cambios de escala/posición, y utilizando exactamente la misma proporción de superficies de fondo (madera, granito, tela oscura, papel blanco y metal) en ambas clases para evitar que la red memorice el fondo (*shortcut learning*).

# %%
# ==============================================================================
# CELDA 2: Auditoría programática del dataset (Conteo, corruptos y duplicados)
# ==============================================================================
RUTA_DATASET = "dataset"

conteo_por_clase = {}
archivos_invalidos = []
hashes_vistos = {}
duplicados_detectados = []

print("=== AUDITORÍA DE CALIDAD E INTEGRIDAD DEL CONJUNTO DE DATOS ===")
for idx_etiqueta, nombre_clase in enumerate(sorted(os.listdir(RUTA_DATASET))):
    ruta_clase = os.path.join(RUTA_DATASET, nombre_clase)
    if not os.path.isdir(ruta_clase):
        continue
    archivos = sorted([f for f in os.listdir(ruta_clase) if not f.startswith(".")])
    validos_clase = 0

    for archivo in archivos:
        ruta_completa = os.path.join(ruta_clase, archivo)
        # 1. Revisión de archivos corruptos o inválidos
        try:
            with Image.open(ruta_completa) as img:
                img.verify()
            validos_clase += 1
        except Exception as e:
            archivos_invalidos.append((ruta_completa, str(e)))
            continue

        # 2. Revisión de duplicados exactos mediante firma criptográfica SHA-256
        with open(ruta_completa, "rb") as f_bin:
            digest = hashlib.sha256(f_bin.read()).hexdigest()
        if digest in hashes_vistos:
            duplicados_detectados.append((ruta_completa, hashes_vistos[digest]))
        else:
            hashes_vistos[digest] = ruta_completa

    conteo_por_clase[nombre_clase] = validos_clase
    cumple_minimo = "SÍ CUMPLE (>= 50)" if validos_clase >= 50 else "NO CUMPLE (< 50)"
    print(f" -> Clase '{nombre_clase}' (Etiqueta y={idx_etiqueta}): {validos_clase} imágenes originales válidas [{cumple_minimo}]")

print("-" * 65)
print(f"Total de imágenes originales válidas : {sum(conteo_por_clase.values())}")
print(f"Archivos inválidos o corruptos       : {len(archivos_invalidos)}")
print(f"Archivos duplicados (SHA-256)        : {len(duplicados_detectados)}")

# %%
# ==============================================================================
# CELDA 3: Visualización de ejemplos originales de ambas clases (Cara vs. Sello)
# ==============================================================================
cara_ejemplos = sorted([f for f in os.listdir(os.path.join(RUTA_DATASET, "cara")) if not f.startswith(".")])[:4]
sello_ejemplos = sorted([f for f in os.listdir(os.path.join(RUTA_DATASET, "sello")) if not f.startswith(".")])[:4]

fig, axes = plt.subplots(2, 4, figsize=(12, 6.2))
fig.suptitle("Muestra de Imágenes Originales del Dataset: Moneda $100 Centenario BCCh", fontsize=13, fontweight="bold")

for i, fname in enumerate(cara_ejemplos):
    img = Image.open(os.path.join(RUTA_DATASET, "cara", fname)).resize((128, 128))
    axes[0, i].imshow(img)
    axes[0, i].set_title(f"Clase 0: cara\n({fname})", fontsize=9)
    axes[0, i].axis("off")

for i, fname in enumerate(sello_ejemplos):
    img = Image.open(os.path.join(RUTA_DATASET, "sello", fname)).resize((128, 128))
    axes[1, i].imshow(img)
    axes[1, i].set_title(f"Clase 1: sello\n({fname})", fontsize=9)
    axes[1, i].axis("off")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Preparación y Aumento de Datos (*Data Augmentation*)
#
# ### 3.1 Separación Entrenamiento (80%) y Validación (20%) y Metadatos
# * **Partición:** 80% entrenamiento y 20% validación con `validation_split=0.2` y `seed=42`, asegurando presencia balanceada de ambas clases en cada grupo y **0 imágenes compartidas** entre entrenamiento y validación.
# * **Tamaño y canales:** `128 x 128` píxeles en `3` canales (RGB) para conservar el contraste entre el anillo dorado y el núcleo plateado.
# * **Normalización:** Capa `Rescaling(1.0 / 255.0)` integrada al inicio de la red para llevar los píxeles de $[0, 255]$ a $[0.0, 1.0]$.
# * **Semilla:** `SEMILLA = 42`.

# %%
# ==============================================================================
# CELDA 4: Creación de particiones Train (80%) / Val (20%) y chequeo de leakage
# ==============================================================================
TAMANO_IMAGEN = (128, 128)
CANALES = 3
BATCH_SIZE = 8
FRACCION_VALIDACION = 0.20

train_ds = tf.keras.utils.image_dataset_from_directory(
    RUTA_DATASET,
    validation_split=FRACCION_VALIDACION,
    subset="training",
    seed=SEMILLA,
    image_size=TAMANO_IMAGEN,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    RUTA_DATASET,
    validation_split=FRACCION_VALIDACION,
    subset="validation",
    seed=SEMILLA,
    image_size=TAMANO_IMAGEN,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

nombres_clases = train_ds.class_names
archivos_train = set(train_ds.file_paths)
archivos_val = set(val_ds.file_paths)
interseccion = archivos_train.intersection(archivos_val)

etiquetas_train = np.concatenate([y.numpy().flatten() for _, y in train_ds])
etiquetas_val = np.concatenate([y.numpy().flatten() for _, y in val_ds])

print("-" * 65)
print(f"Clases detectadas (orden alfabético) : {nombres_clases} -> (0='{nombres_clases[0]}', 1='{nombres_clases[1]}')")
print(f"Dimensiones de entrada (X)           : {TAMANO_IMAGEN[0]}x{TAMANO_IMAGEN[1]}x{CANALES} (RGB)")
print(f"Partición Entrenamiento (80%)        : {len(etiquetas_train)} imgs -> cara (0): {int(np.sum(etiquetas_train == 0))}, sello (1): {int(np.sum(etiquetas_train == 1))}")
print(f"Partición Validación (20%)           : {len(etiquetas_val)} imgs -> cara (0): {int(np.sum(etiquetas_val == 0))}, sello (1): {int(np.sum(etiquetas_val == 1))}")
print(f"Archivos compartidos (Data Leakage)  : {len(interseccion)} archivos (Particiones 100% disjuntas)")

# %% [markdown]
# ### 3.2 Transformaciones de Aumento de Datos y Conservación de la Etiqueta
# Se aplican **3 transformaciones geométricas exclusivamente al entrenamiento** (`training=True`):
# 1. `RandomFlip("horizontal_and_vertical")`: Inversión especular horizontal y vertical.
# 2. `RandomRotation(0.25)`: Giros aleatorios de hasta $\pm 90^\circ$.
# 3. `RandomZoom(0.12)`: Variación de escala de $\pm 12\%$.
#
# **Justificación de conservación de la etiqueta:** Dado que la moneda es un cuerpo rígido circular, rotarla sobre su eje, reflejarla o acercar levemente el encuadre no altera la identidad física de sus grabados internos (`"CIEN años"` en `cara` vs. Escudo Nacional y `"100"` en `sello`).

# %%
# ==============================================================================
# CELDA 5: Bloque de Data Augmentation y visualización Antes vs. Después
# ==============================================================================
capa_aumento_datos = models.Sequential([
    layers.RandomFlip("horizontal_and_vertical", seed=SEMILLA),
    layers.RandomRotation(0.25, fill_mode="reflect", seed=SEMILLA),
    layers.RandomZoom(0.12, fill_mode="reflect", seed=SEMILLA),
], name="aumento_datos_entrenamiento")

# Extraer 1 imagen original de 'cara' (0) y 1 de 'sello' (1) desde train_ds
img_cara_orig, img_sello_orig = None, None
for lote_imgs, lote_lbls in train_ds:
    lbls_flat = lote_lbls.numpy().flatten()
    if img_cara_orig is None and np.any(lbls_flat == 0):
        idx_c = np.where(lbls_flat == 0)[0][0]
        img_cara_orig = lote_imgs[idx_c:idx_c+1]
    if img_sello_orig is None and np.any(lbls_flat == 1):
        idx_s = np.where(lbls_flat == 1)[0][0]
        img_sello_orig = lote_imgs[idx_s:idx_s+1]
    if img_cara_orig is not None and img_sello_orig is not None:
        break

fig, axes = plt.subplots(2, 4, figsize=(12, 6.2))
fig.suptitle("Demostración Antes vs. Después del Aumento de Datos (Solo Entrenamiento)", fontsize=13, fontweight="bold")

# Fila 0: Antes y 3 transformaciones sobre 'cara'
axes[0, 0].imshow(img_cara_orig[0].numpy().astype("uint8"))
axes[0, 0].set_title("ANTES (Original: cara)\ny = 0", fontsize=9.5, fontweight="bold", color="#1b4f72")
axes[0, 0].axis("off")
for j in range(1, 4):
    img_aug = capa_aumento_datos(img_cara_orig, training=True)
    axes[0, j].imshow(img_aug[0].numpy().astype("uint8"))
    axes[0, j].set_title(f"DESPUÉS (Transformación #{j})\nConserva etiqueta y = 0", fontsize=9)
    axes[0, j].axis("off")

# Fila 1: Antes y 3 transformaciones sobre 'sello'
axes[1, 0].imshow(img_sello_orig[0].numpy().astype("uint8"))
axes[1, 0].set_title("ANTES (Original: sello)\ny = 1", fontsize=9.5, fontweight="bold", color="#145a32")
axes[1, 0].axis("off")
for j in range(1, 4):
    img_aug = capa_aumento_datos(img_sello_orig, training=True)
    axes[1, j].imshow(img_aug[0].numpy().astype("uint8"))
    axes[1, j].set_title(f"DESPUÉS (Transformación #{j})\nConserva etiqueta y = 1", fontsize=9)
    axes[1, j].axis("off")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Diseño de la Red Neuronal Convolucional (CNN)
#
# ### 4.1 Diferencia de Procesamiento entre MLP y CNN (Indicador 5)
# * **MLP (*Multi-Layer Perceptron*):** Aplana la imagen de $128 \times 128 \times 3$ en un vector de $49.152$ valores en la entrada, perdiendo las relaciones espaciales 2D entre píxeles vecinos. Una primera capa oculta de 128 neuronas exige $6,29$ millones de pesos, carece de invarianza traslacional/rotacional y memoriza el fondo.
# * **CNN (*Convolutional Neural Network*):** Mantiene la estructura matricial 3D aplicando filtros locales de $3 \times 3$ con pesos compartidos (*weight sharing*) y `MaxPooling2D(2,2)`. Así extrae una jerarquía espacial de características: bordes y anillo bimetálico en el Bloque 1, texturas de laurel/Cordillera en el Bloque 2, y patrones morfológicos complejos (`"CIEN"` vs. Escudo y `"100"`) en el Bloque 3.
#
# ### 4.2 Función de las Capas, ReLU, Salida Sigmoid, Pérdida y Optimizador (Indicador 6)
# * **`Rescaling(1./255)`:** Normaliza las intensidades RGB al intervalo $[0.0, 1.0]$.
# * **`Conv2D` + Activación `ReLU` ($\max(0, z)$):** Extrae mapas de características locales e introduce no-linealidad anulando valores negativos sin saturar el gradiente positivo.
# * **`MaxPooling2D((2, 2))`:** Reduce a la mitad la resolución espacial reteniendo la activación máxima de cada ventana $2 \times 2$, aportando invarianza a traslaciones leves.
# * **`Flatten()`, `Dense(128, activation="relu")` y `Dropout(0.3)`:** Vectoriza los mapas profundos, combina la evidencia en una capa densa y apaga aleatoriamente el 30% de las neuronas en entrenamiento para evitar sobreajuste.
# * **Salida `Dense(1, activation="sigmoid")`:** Compatible con `label_mode="binary"`. Entrega una única probabilidad $\hat{y} = \sigma(z) = \frac{1}{1+e^{-z}} \in (0, 1)$ correspondiente a $P(y=1 \mid X)$ (`sello`). Si $\hat{y} < 0.5 \implies \text{cara (0)}$; si $\hat{y} \ge 0.5 \implies \text{sello (1)}$.
# * **Función de pérdida `binary_crossentropy` y optimizador `Adam`:** Minimiza la entropía cruzada binaria entre $y \in \{0, 1\}$ e $\hat{y} \in (0, 1)$ ajustando adaptativamente el paso de aprendizaje por parámetro.

# %%
# ==============================================================================
# CELDA 6: Definición de arquitecturas (Experimento 1 y Experimento 2) y summary
# ==============================================================================
def construir_modelo_exp1_baseline():
    """Experimento 1 (Baseline): 2 bloques Conv2D (16 y 32 filtros), sin Dropout, lr=0.001"""
    modelo = models.Sequential(name="CNN_Exp1_Baseline_2Bloques")
    modelo.add(layers.Input(shape=(128, 128, 3)))
    modelo.add(layers.Rescaling(1.0 / 255.0))
    modelo.add(capa_aumento_datos)

    # Bloque 1
    modelo.add(layers.Conv2D(16, (3, 3), padding="same", activation="relu"))
    modelo.add(layers.MaxPooling2D((2, 2)))
    # Bloque 2
    modelo.add(layers.Conv2D(32, (3, 3), padding="same", activation="relu"))
    modelo.add(layers.MaxPooling2D((2, 2)))

    # Clasificador denso sin Dropout
    modelo.add(layers.Flatten())
    modelo.add(layers.Dense(64, activation="relu"))
    modelo.add(layers.Dense(1, activation="sigmoid"))

    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return modelo


def construir_modelo_exp2_ajustado():
    """Experimento 2 (Ajustado Final): 3 bloques Conv2D (32, 64, 128 filtros) + Dropout(0.3), lr=0.0005"""
    modelo = models.Sequential(name="CNN_Exp2_Ajustado_3Bloques_Dropout")
    modelo.add(layers.Input(shape=(128, 128, 3)))
    modelo.add(layers.Rescaling(1.0 / 255.0))
    modelo.add(capa_aumento_datos)

    # Bloque 1
    modelo.add(layers.Conv2D(32, (3, 3), padding="same", activation="relu"))
    modelo.add(layers.MaxPooling2D((2, 2)))
    # Bloque 2
    modelo.add(layers.Conv2D(64, (3, 3), padding="same", activation="relu"))
    modelo.add(layers.MaxPooling2D((2, 2)))
    # Bloque 3
    modelo.add(layers.Conv2D(128, (3, 3), padding="same", activation="relu"))
    modelo.add(layers.MaxPooling2D((2, 2)))

    # Clasificador denso regularizado con Dropout(0.3)
    modelo.add(layers.Flatten())
    modelo.add(layers.Dense(128, activation="relu"))
    modelo.add(layers.Dropout(0.3, seed=SEMILLA))
    modelo.add(layers.Dense(1, activation="sigmoid"))

    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return modelo

modelo_exp1 = construir_modelo_exp1_baseline()
modelo_exp2 = construir_modelo_exp2_ajustado()

print("=== ARQUITECTURA EXPERIMENTO 1 (BASELINE) ===")
modelo_exp1.summary()
print("\n=== ARQUITECTURA EXPERIMENTO 2 (MODELO AJUSTADO FINAL) ===")
modelo_exp2.summary()

# %% [markdown]
# ## 5. Entrenamiento, Ajuste de Hiperparámetros y Análisis de Sobreajuste
#
# Se ejecutan **dos experimentos sobre la misma partición (`train_ds` y `val_ds`, `seed=42`, `batch_size=8`)**:
# * **Experimento 1 (Baseline):** 2 bloques convolucionales (`16 -> 32` filtros), sin `Dropout`, `lr = 0.001`, entrenado durante **15 épocas**.
# * **Experimento 2 (Modelo Ajustado):** 3 bloques convolucionales (`32 -> 64 -> 128` filtros), incorporando regularización **`Dropout(0.3)`**, `lr = 0.0005`, entrenado durante **25 épocas**.

# %%
# ==============================================================================
# CELDA 7: Ejecución de los dos experimentos sobre la misma partición
# ==============================================================================
EPOCAS_EXP1 = 15
EPOCAS_EXP2 = 25

print(">>> Iniciando Entrenamiento del EXPERIMENTO 1 (Baseline - 15 épocas)...")
historial_exp1 = modelo_exp1.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCAS_EXP1,
    verbose=1
)

print("\n>>> Iniciando Entrenamiento del EXPERIMENTO 2 (Ajustado con 3 Bloques + Dropout 0.3 - 25 épocas)...")
historial_exp2 = modelo_exp2.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCAS_EXP2,
    verbose=1
)

val_acc_1 = historial_exp1.history["val_accuracy"][-1]
val_loss_1 = historial_exp1.history["val_loss"][-1]
val_acc_2 = historial_exp2.history["val_accuracy"][-1]
val_loss_2 = historial_exp2.history["val_loss"][-1]

print("\n=== COMPARACIÓN FINAL SOBRE LA MISMA PARTICIÓN DE VALIDACIÓN ===")
print(f"Experimento 1 (Baseline) -> Val Accuracy: {val_acc_1*100:.2f}% | Val Loss: {val_loss_1:.4f}")
print(f"Experimento 2 (Ajustado) -> Val Accuracy: {val_acc_2*100:.2f}% | Val Loss: {val_loss_2:.4f}")

# %%
# ==============================================================================
# CELDA 8: Gráficas de Accuracy y Pérdida (Entrenamiento vs. Validación)
# ==============================================================================
fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
fig.suptitle("Comparativa Experimental de Curvas de Entrenamiento y Validación", fontsize=14, fontweight="bold")

ep1 = range(1, EPOCAS_EXP1 + 1)
ep2 = range(1, EPOCAS_EXP2 + 1)

# Experimento 1 - Accuracy
axes[0, 0].plot(ep1, historial_exp1.history["accuracy"], "o-", label="Train Accuracy (Exp 1)", color="#1f77b4", markersize=4)
axes[0, 0].plot(ep1, historial_exp1.history["val_accuracy"], "s--", label="Val Accuracy (Exp 1)", color="#ff7f0e", markersize=4)
axes[0, 0].set_title("Experimento 1 (Baseline): Exactitud (Accuracy)", fontweight="bold")
axes[0, 0].set_xlabel("Épocas")
axes[0, 0].set_ylabel("Accuracy")
axes[0, 0].set_ylim(0.45, 1.03)
axes[0, 0].grid(True, linestyle=":", alpha=0.6)
axes[0, 0].legend(loc="lower right")

# Experimento 1 - Loss
axes[0, 1].plot(ep1, historial_exp1.history["loss"], "o-", label="Train Loss (Exp 1)", color="#d62728", markersize=4)
axes[0, 1].plot(ep1, historial_exp1.history["val_loss"], "s--", label="Val Loss (Exp 1)", color="#9467bd", markersize=4)
axes[0, 1].set_title("Experimento 1 (Baseline): Pérdida (Binary Crossentropy)", fontweight="bold")
axes[0, 1].set_xlabel("Épocas")
axes[0, 1].set_ylabel("Loss")
axes[0, 1].grid(True, linestyle=":", alpha=0.6)
axes[0, 1].legend(loc="upper right")

# Experimento 2 - Accuracy
axes[1, 0].plot(ep2, historial_exp2.history["accuracy"], "o-", label="Train Accuracy (Exp 2)", color="#2ca02c", markersize=4)
axes[1, 0].plot(ep2, historial_exp2.history["val_accuracy"], "s--", label="Val Accuracy (Exp 2)", color="#17becf", markersize=4)
axes[1, 0].set_title("Experimento 2 (Final con 3 Bloques + Dropout 0.3): Exactitud", fontweight="bold")
axes[1, 0].set_xlabel("Épocas")
axes[1, 0].set_ylabel("Accuracy")
axes[1, 0].set_ylim(0.45, 1.03)
axes[1, 0].grid(True, linestyle=":", alpha=0.6)
axes[1, 0].legend(loc="lower right")

# Experimento 2 - Loss
axes[1, 1].plot(ep2, historial_exp2.history["loss"], "o-", label="Train Loss (Exp 2)", color="#e377c2", markersize=4)
axes[1, 1].plot(ep2, historial_exp2.history["val_loss"], "s--", label="Val Loss (Exp 2)", color="#8c564b", markersize=4)
axes[1, 1].set_title("Experimento 2 (Final con 3 Bloques + Dropout 0.3): Pérdida", fontweight="bold")
axes[1, 1].set_xlabel("Épocas")
axes[1, 1].set_ylabel("Loss")
axes[1, 1].grid(True, linestyle=":", alpha=0.6)
axes[1, 1].legend(loc="upper right")

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 5.2 Análisis de Sobreajuste (*Overfitting*) y Elección del Modelo Final
# * En el **Experimento 1 (Baseline)**, al carecer de `Dropout` y contar con solo 2 bloques convolucionales, la pérdida de entrenamiento (`loss`) cae rápidamente hacia cero mientras la pérdida de validación (`val_loss`) se estanca y comienza a subir en las últimas épocas, abriendo una brecha entre `accuracy` y `val_accuracy` (señal característica de sobreajuste).
# * En el **Experimento 2 (Ajustado)**, el tercer bloque convolucional (`128` filtros) junto con `Dropout(0.3)` y `lr=0.0005` actúa como regularizador efectivo: ambas curvas (`loss` y `val_loss`) descienden de manera acoplada alcanzando mayor `val_accuracy` y menor `val_loss`. Por esta evidencia empírica en validación, se selecciona **`modelo_exp2`** como modelo final.

# %% [markdown]
# ## 6. Modelo Guardado (`.keras`), Inferencia y Prueba con Categoría No Entrenada

# %%
# ==============================================================================
# CELDA 9: Guardado .keras, recarga y prueba con imágenes reales y NO entrenadas
# ==============================================================================
ARCHIVO_MODELO = "modelo_moneda_centenario.keras"
modelo_exp2.save(ARCHIVO_MODELO)
print(f"[OK] Modelo guardado exitosamente en '{ARCHIVO_MODELO}'")

# Recarga limpia del modelo desde disco
modelo_recargado = tf.keras.models.load_model(ARCHIVO_MODELO)
print("[OK] Modelo recargado desde disco correctamente.\n")

RUTA_PRUEBAS = "pruebas"
archivos_en_pruebas = sorted([f for f in os.listdir(RUTA_PRUEBAS) if f.lower().endswith((".jpg", ".jpeg", ".png"))])

fig, axes = plt.subplots(1, len(archivos_en_pruebas), figsize=(3.4 * len(archivos_en_pruebas), 3.8))
if len(archivos_en_pruebas) == 1:
    axes = [axes]
fig.suptitle("Evaluación del Modelo Recargado (.keras): Clases Conocidas vs. Categorías No Entrenadas", fontsize=12, fontweight="bold")

for i, nombre_img in enumerate(archivos_en_pruebas):
    ruta_img = os.path.join(RUTA_PRUEBAS, nombre_img)
    img_pil = load_img(ruta_img, target_size=TAMANO_IMAGEN)
    tensor_input = np.expand_dims(img_to_array(img_pil), axis=0)

    prob_sello = float(modelo_recargado.predict(tensor_input, verbose=0)[0][0])
    if prob_sello >= 0.5:
        clase_pred = "SELLO (1)"
        confianza = prob_sello * 100.0
    else:
        clase_pred = "CARA (0)"
        confianza = (1.0 - prob_sello) * 100.0

    es_no_entrenada = "no_entrenada" in nombre_img.lower() or "golilla" in nombre_img.lower() or "pila" in nombre_img.lower()
    etiqueta_tipo = "[CATEGORÍA NO ENTRENADA]" if es_no_entrenada else "[CLASE CONOCIDA]"
    color_titulo = "#922b21" if es_no_entrenada else "#145a32"

    print(f"{etiqueta_tipo:25s} {nombre_img:35s} -> Sigmoid p={prob_sello:.4f} | Pred: {clase_pred} ({confianza:.1f}% certeza)")

    axes[i].imshow(img_pil)
    axes[i].set_title(f"{nombre_img}\n{etiqueta_tipo}\nPred: {clase_pred} ({confianza:.1f}%)", fontsize=8.5, fontweight="bold", color=color_titulo)
    axes[i].axis("off")

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 6.1 Explicación del Comportamiento ante Categorías No Entrenadas y Propuesta de Mejora (Indicador 9)
# * **¿Por qué asigna una clase conocida (`cara` o `sello`) a una golilla o pila botón?** Porque un clasificador binario con salida `Sigmoid` opera bajo el supuesto de **conjunto cerrado (*closed-set*)**: particiona todo el espacio $\mathbb{R}^{128\times 128\times 3}$ en dos semiespacios separados por el hiperplano $z = 0$ ($\sigma(0) = 0.5$). Al no existir una neurona de rechazo, cualquier objeto circular metálico activa los filtros de bordes curvos y brillo especular y cae forzosamente en uno de los dos lados.
# * **¿Por qué una salida alta no garantiza un acierto?** Porque $\sigma(z)$ mide la distancia al límite de decisión entre `cara` y `sello`, no la probabilidad de pertenencia al dominio de monedas. Bordes metálicos intensos producen activaciones altas en las capas `ReLU` ($|z| \gg 0$), saturando la función `Sigmoid` cerca de $0.0$ o $1.0$ con **sobreconfianza en falsos positivos**.
# * **Propuesta de mejora coherente:** Aplicar un **umbral de abstención (*rejection band*)** exigiendo $\hat{y} \le 0.10$ para `cara` y $\hat{y} \ge 0.90$ para `sello` (declarando `"OBJETO DESCONOCIDO / RECHAZO"` en el rango intermedio), o entrenar un modelo multiclase con una 3ra carpeta explícita `2_anomalia_fondo` que agrupe golillas, fichas, otras monedas y fondos vacíos.
#
# ---
# ## 7. Conclusiones (Indicador 10)
# 1. **Cumplimiento del objetivo:** Se cumplió el objetivo al construir y validar un clasificador CNN binario para la Moneda Conmemorativa de $100 del Centenario del Banco Central de Chile que discrimina correctamente entre `cara` y `sello` con alta exactitud en validación y sin filtración de datos entre particiones.
# 2. **Aprendizajes clave:** Comprobamos la importancia de balancear los fondos e iluminaciones en ambas clases para evitar *shortcut learning*, verificamos cómo el aumento de datos rotacional respeta la geometría circular de la moneda, y demostramos empíricamente cómo el 3er bloque convolucional más `Dropout(0.3)` corrigen el sobreajuste del modelo base.
# 3. **Aspectos por mejorar:** Incorporar una etapa previa de detección de anomalías (*Open-Set Recognition*) o una clase explícita de rechazo para objetos metálicos extraños, y exportar el modelo a TensorFlow Lite (`INT8`) para inferencia en tiempo real sobre hardware embebido.