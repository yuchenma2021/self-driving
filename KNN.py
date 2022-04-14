import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA


training_images = np.array(np.load('train_data.npy', allow_pickle=True))
training_labels = np.array(np.load('train_labels.npy', allow_pickle=True))
validation_images = np.array(np.load('valid_data.npy', allow_pickle=True))
validation_labels = np.array(np.load('valid_labels.npy', allow_pickle=True))
testing_images = np.array(np.load('test_data.npy', allow_pickle=True))
testing_labels = np.array(np.load('test_labels.npy', allow_pickle=True))


scaler = StandardScaler()
training_images = scaler.fit_transform(training_images)
testing_images = scaler.transform(testing_images)

n_components = 2
pca = PCA(n_components=n_components)
training_images = pca.fit_transform(training_images)
testing_images = pca.fit_transform(testing_images)
clf = KNeighborsClassifier(n_neighbors=11).fit(training_images, training_labels)
yhat = clf.predict(testing_images)

accuracy = accuracy_score(yhat,testing_labels)
print(accuracy)

#0.6731700370264881
