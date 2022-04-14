import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pickle
training_images = np.array(np.load('train_data.npy', allow_pickle=True))
training_labels = np.array(np.load('train_labels.npy', allow_pickle=True))
validation_images = np.array(np.load('valid_data.npy', allow_pickle=True))
validation_labels = np.array(np.load('valid_labels.npy', allow_pickle=True))
testing_images = np.array(np.load('test_data.npy', allow_pickle=True))
testing_labels = np.array(np.load('test_labels.npy', allow_pickle=True))

scaler = StandardScaler()
training_images = scaler.fit_transform(training_images)
testing_images = scaler.transform(testing_images)

model = LogisticRegression(multi_class='ovr')
model.fit(training_images, training_labels)
yhat = model.predict(testing_images)
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))
accuracy = accuracy_score(yhat,testing_labels)
print(accuracy)

#0.799
