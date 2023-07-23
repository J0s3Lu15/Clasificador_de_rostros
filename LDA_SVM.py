from sklearn.datasets import fetch_olivetti_faces
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# Obtener el dataset

faces = fetch_olivetti_faces()

_, img_height, img_width = faces.images.shape

print(faces.images.shape)

# Dividir el dataset

N_IDENTITIES = len(np.unique(faces.target)) # cantidad de individuos
GALLERY_SIZE = 5                            # imagenes para entrenamiento

gallery_indices = []
probe_indices = []
for i in range(N_IDENTITIES):
    indices = list(np.where(faces.target == i)[0])
    gallery_indices += indices[:GALLERY_SIZE]
    probe_indices += indices[GALLERY_SIZE:]

x_train = faces.images[gallery_indices].reshape(-1, img_height*img_width) # vectorize train images
y_train = faces.target[gallery_indices]
x_test = faces.images[probe_indices].reshape(-1, img_height*img_width)    # vectorize test images
y_test = faces.target[probe_indices]

print(x_train.shape, x_test.shape)

# Visualizar el dataset dividido
def show_images(imgs, num_rows, num_cols):
    assert len(imgs) == num_rows*num_cols

    full = None
    for i in range(num_rows):
        row = None
        for j in range(num_cols):
            if row is None:
                row = imgs[i*num_cols+j].reshape(img_height, img_width)*255.0
            else:
                row = np.concatenate((row, imgs[i*num_cols+j].reshape(img_height, img_width)*255.0), axis=1)
        if full is None:
            full = row
        else:
            full = np.concatenate((full, row), axis=0)

    f = plt.figure(figsize=(num_cols, num_rows))
    plt.imshow(full, cmap='gray')
    plt.axis('off')
    plt.show()

print('TRAINING')
show_images(x_train, N_IDENTITIES, GALLERY_SIZE)
print('TESTING')
show_images(x_test, N_IDENTITIES, 10 - GALLERY_SIZE)


# Aplicar LDA
lda = LDA(n_components=2)
x_train_lda = lda.fit_transform(x_train, y_train)
x_test_lda = lda.transform(x_test)

# Visualización de la representación 2D de las caras calculada por LDA
plt.figure(figsize=(8, 6))
for i in range(N_IDENTITIES):
    plt.scatter(x_train_lda[y_train == i, 0], x_train_lda[y_train == i, 1], label=f"Individuo {i}")
plt.xlabel("Componente 1")
plt.ylabel("Componente 2")
plt.title("Visualización 2D de las caras utilizando LDA")
plt.legend()
plt.show()

# F-Score promedio para LDA
f1_avg_lda = f1_score(y_test, lda.predict(x_test), average='macro')

# Matriz de confusión para LDA
confusion_mat_lda = confusion_matrix(y_test, lda.predict(x_test))

# Individuos con mayor confusión para LDA
confused_individuals_lda = []
for i in range(N_IDENTITIES):
    if np.sum(confusion_mat_lda[i, :]) > 0:
        confused_individuals_lda.append(i)

print("LDA:")
print("F-Score promedio:", f1_avg_lda)
print("Matriz de confusión:")
print(confusion_mat_lda)
print("Individuos con mayor confusión:", confused_individuals_lda)


# Entrenamiento y evaluación de SVM
svm = SVC()
svm.fit(x_train, y_train)  # Entrenamiento de SVM

# Predicciones en el conjunto de prueba
y_pred = svm.predict(x_test)

# Cálculo del F-Score promedio
f1_avg = f1_score(y_test, y_pred, average='macro')

# Cálculo de la matriz de confusión
confusion_mat = confusion_matrix(y_test, y_pred)

# Visualización de individuos con mayor confusión
confused_individuals = []
for i in range(N_IDENTITIES):
    if np.sum(confusion_mat[i, :]) > 0:
        confused_individuals.append(i)

print("SVM:")
print("F-Score promedio:", f1_avg)
print("Matriz de confusión:")
print(confusion_mat)
print("Individuos con mayor confusión:", confused_individuals)