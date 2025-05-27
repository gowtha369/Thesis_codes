import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import joblib

train_healthy = "E:/plants_data/train/healthy"
train_diseased = "E:/plants_data/train/diseased"
test_healthy = "E:/plants_data/test/healthy"
test_diseased = "E:/plants_data/test/diseased"

def load_images_svm(folder_path, label, img_size=(64, 64)):
    data = []
    for img_file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_file)
        try:
            img = load_img(img_path, target_size=img_size)
            img_array = img_to_array(img)
            data.append((img_array.flatten(), label))
        except:
            print(f"Error loading {img_path}")
    return data


svm_data = load_images_svm(train_healthy, "healthy") + \
           load_images_svm(train_diseased, "diseased") + \
           load_images_svm(test_healthy, "healthy") + \
           load_images_svm(test_diseased, "diseased")


X, y = zip(*svm_data)
X = np.array(X)
y = np.array(y)


le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

svm_model = SVC(kernel='linear', C=1)
svm_model.fit(X_train, y_train)

joblib.dump(svm_model, 'svm_model.joblib')
joblib.dump(le, 'label_encoder.joblib')
print("Model and label encoder saved successfully!")

svm_model = joblib.load('svm_model.joblib')
le = joblib.load('label_encoder.joblib')

y_pred = svm_model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

def predict_single_image(image_path, model, label_encoder, img_size=(64, 64)):
    try:
        img = load_img(image_path, target_size=img_size)
        img_array = img_to_array(img)
        img_flattened = img_array.flatten().reshape(1, -1)
        prediction = model.predict(img_flattened)
        predicted_label = label_encoder.inverse_transform(prediction)
        return predicted_label[0]
    except Exception as e:
        return f"Error: {e}"

image_path = "E:/test.jpg"  
result = predict_single_image(image_path, svm_model, le)
print("Predicted class:", result)


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


cm = confusion_matrix(y_test, y_pred)


disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)


disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

