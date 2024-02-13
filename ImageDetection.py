import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

# Load pre-trained ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False)

# Function to extract features from images using ResNet50
def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = base_model.predict(x)
    return features.flatten()

# Example data: paths to authentic and manipulated images
authentic_image_paths = ["authentic_image1.jpg", "authentic_image2.jpg", ...]
manipulated_image_paths = ["manipulated_image1.jpg", "manipulated_image2.jpg", ...]

# Extract features from images
authentic_features = np.array([extract_features(path) for path in authentic_image_paths])
manipulated_features = np.array([extract_features(path) for path in manipulated_image_paths])

# Create labels (0 for authentic, 1 for manipulated)
authentic_labels = np.zeros(authentic_features.shape[0])
manipulated_labels = np.ones(manipulated_features.shape[0])

# Combine data and labels
X = np.concatenate([authentic_features, manipulated_features], axis=0)
y = np.concatenate([authentic_labels, manipulated_labels], axis=0)

# Train SVM classifier
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X, y)

# Evaluate model
y_pred = svm_classifier.predict(X)
accuracy = accuracy_score(y, y_pred)
print("Accuracy:", accuracy)
