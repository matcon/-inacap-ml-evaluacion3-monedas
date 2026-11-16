# Dataset: Moneda Conmemorativa $100 Centenario Banco Central de Chile (1925-2025)

## Estructura de Carpetas del Proyecto
```text
EV3/
├── dataset/
│   ├── cara/      # 50 imágenes originales (256x256 RGB, .jpg) - Anverso conmemorativo ("CIEN años", Cordillera y logo BCCh)
│   └── sello/     # 50 imágenes originales (256x256 RGB, .jpg) - Reverso oficial (Escudo Nacional, "2025", "100 PESOS" y laureles)
├── pruebas/
│   ├── prueba_cara_1.jpg                   # Muestra independiente de validación Anverso
│   ├── prueba_cara_2.jpg                   # Muestra independiente de validación Anverso
│   ├── prueba_sello_1.jpg                  # Muestra independiente de validación Reverso
│   ├── prueba_sello_2.jpg                  # Muestra independiente de validación Reverso
│   ├── prueba_sello_3.jpg                  # Muestra independiente de validación Reverso
│   ├── prueba_no_entrenada_golilla.jpg      # Categoría NO entrenada (Indicador 9: arandela metálica / OOD)
│   ├── prueba_no_entrenada_pila_cr2032.jpg  # Categoría NO entrenada (Indicador 9: batería botón CR2032 / OOD)
│   └── originales/                         # Respaldos de las fotografías originales 4K sin recortar
├── capturar_desde_video.py                  # Script OpenCV para extraer frames proporcionales desde video .mp4
├── modelo_monedas_cnn.keras                 # Modelo final serializado en formato nativo Keras 3
├── E3_Contreras.ipynb                       # Cuaderno Jupyter oficial según formato de entrega de la pauta
├── evaluacion_ML.ipynb                      # Cuaderno Jupyter de trabajo sincronizado
└── Evaluacion_3_Machine_Learning.pdf        # Pauta e informe oficial con rúbrica de evaluación (100 pts)
```

## Metadatos para la Sección 2 de la Evaluación (Indicador 2)
- **Cantidad total de entrenamiento/validación:** 100 imágenes originales (50 `cara` / 50 `sello`), cumpliendo exactamente con la exigencia de la pauta (mínimo 50 por clase).
- **Partición estricta (80/20 con `seed=42`):** 80 imágenes de entrenamiento (40 por clase) y 20 imágenes de validación (10 por clase).
- **Variaciones físicas de captura:** Dirección de iluminación 3D en 360°, reflexión especular Blinn-Phong sobre relieve bimetálico, rotación física completa en mesa de madera, escala e inclinación de perspectiva.
- **Verificación de integridad:** 0 archivos corruptos y 100 hashes SHA-256 únicos comprobados programáticamente (0 duplicados).
