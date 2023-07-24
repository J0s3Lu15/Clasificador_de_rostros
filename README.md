# Reconocimiento Facial usando Análisis Discriminante Lineal (LDA) y Máquinas de Soporte Vectorial (SVM)

Este repositorio contiene código en Python para realizar reconocimiento facial usando Análisis Discriminante Lineal (LDA) y Máquinas de Soporte Vectorial (SVM) sobre el conjunto de datos Olivetti Faces. El conjunto de datos Olivetti Faces consiste en imágenes en escala de grises de 40 individuos, cada una capturada bajo diversas condiciones de iluminación y expresiones faciales.

## Autor:
José Luis Flores Tito - Analista de Ciberseguridad

## Dataset

El conjunto de datos Olivetti Faces se descargará e importará automáticamente usando la función `fetch_olivetti_faces` del módulo `sklearn.datasets`. Este conjunto de datos contiene 400 imágenes de tamaño 64x64, que representan a 40 individuos (10 imágenes por individuo).

## Instrucciones

Instala las bibliotecas necesarias usando el siguiente comando:
```bash
pip install numpy matplotlib scikit-learn
```
Ejecuta el script de reconocimiento facial
```bash
python LDA_SVM.py
```

## Reconocimiento Facial usando LDA
## Resumen
Se utiliza el Análisis Discriminante Lineal (LDA) para proyectar las imágenes faciales de alta dimensión en un espacio de menor dimensión, maximizando la separabilidad entre las clases. Esta transformación permite visualizar las representaciones faciales en un espacio 2D.

## Visualización de la Representación 2D de las Caras
El script mostrará un gráfico de dispersión 2D que muestra la visualización de las imágenes faciales utilizando LDA. Cada punto en el gráfico representa una imagen facial, y los colores diferentes indican diferentes individuos.

## Evaluación
El rendimiento del reconocimiento facial basado en LDA se evalúa utilizando el F-Score y una matriz de confusión. El F-Score proporciona una medida general de precisión, y la matriz de confusión muestra el número de predicciones correctas e incorrectas para cada individuo.

## Reconocimiento Facial usando SVM
## Resumen
Se emplean las Máquinas de Soporte Vectorial (SVM) para el reconocimiento facial. SVM es un potente algoritmo de clasificación que encuentra un hiperplano óptimo para separar diferentes clases faciales en el espacio de características de alta dimensión.

## Evaluación
Al igual que con LDA, el script calculará el F-Score y la matriz de confusión para evaluar el rendimiento del reconocimiento facial basado en SVM. Además, se identificarán individuos con mayor confusión, lo que indica posibles desafíos en el reconocimiento.

## Contactos:
Si te gusta mi trabajo o estás buscando consultoría para tus proyectos, Pentesting, servicios de RED TEAM - BLUE TEAM, implementación de normas de seguridad e ISOs, controles IDS - IPS, gestión de SIEM, implementación de topologías de red seguras, entrenamiento e implementación de modelos de IA, desarrollo de sistemas, Apps Móviles, Diseño Gráfico, Marketing Digital y todo lo relacionado con la tecnología, no dudes en contactarme al +591 75764248 y con gusto trabajare contigo.
