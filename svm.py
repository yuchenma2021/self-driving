#%%
import numpy as np
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
#%%


training_images = np.array(np.load('train_data.npy', allow_pickle=True))
training_labels = np.array(np.load('train_labels.npy', allow_pickle=True))
testing_images = np.array(np.load('test_data.npy', allow_pickle=True))
testing_labels = np.array(np.load('test_labels.npy', allow_pickle=True))

scaler = StandardScaler()
training_images = scaler.fit_transform(training_images)
testing_images = scaler.transform(testing_images)


###PCA
n_components = 2
pca = PCA(n_components=n_components)
training_images = pca.fit_transform(training_images)
testing_images = pca.fit_transform(testing_images)


svc=svm.SVC(probability=True,max_iter=1000,verbose=True)
model = svc
model.fit(training_images,training_labels)
y_pred=model.predict(testing_images)


accuracy = accuracy_score(y_pred,testing_labels)
print(accuracy)

#0.5684990031330105