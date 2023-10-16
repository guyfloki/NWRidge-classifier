from models.nadaraya_watson_ridge_classifier import AdamWOptimizer
from models.nadaraya_watson_ridge_classifier import NadarayaWatsonRidgeClassifer
from utils.data_generation import generate_multiclass_data
from utils.model_comparison import compare_models
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time
import matplotlib.pyplot as plt 
import numpy as np

# Data Generation
X_train_multiclass, X_test_multiclass, y_train_multiclass, y_test_multiclass = generate_multiclass_data()

# Model Initialization
nw_multi_classifier = NadarayaWatsonRidgeClassifer(alpha=1.0, h=2.0, batch_size=200)
svc_classifier = SVC(probability=True)

# Training and Timing NadarayaWatsonRidgeClassifer
start_time_nw = time.time()
nw_multi_classifier.fit(X_train_multiclass, y_train_multiclass)
end_time_nw = time.time()
nw_training_time = end_time_nw - start_time_nw

# Training and Timing SVC
start_time_svc = time.time()
svc_classifier.fit(X_train_multiclass, y_train_multiclass)
end_time_svc = time.time()
svc_training_time = end_time_svc - start_time_svc

# Predictions
y_pred_nw = nw_multi_classifier.predict(X_test_multiclass)
y_pred_svc = svc_classifier.predict(X_test_multiclass)

# Accuracy Calculation
accuracy_nw = accuracy_score(y_test_multiclass, y_pred_nw)
accuracy_svc = accuracy_score(y_test_multiclass, y_pred_svc)

print(f"NWRC Training Time: {nw_training_time:.4f} seconds")
print(f"SVC Training Time: {svc_training_time:.4f} seconds")
print(f"NWRC Accuracy: {accuracy_nw:.4f}")
print(f"SVC Accuracy: {accuracy_svc:.4f}")

# Comparing the models
models_info = [
    {"name": "NWRC", "train_time": nw_training_time, "accuracy": accuracy_nw},
    {"name": "SVC", "train_time": svc_training_time, "accuracy": accuracy_svc}
]

classifiers = {
    "NWRC": nw_multi_classifier,
    "SVC": svc_classifier
}

compare_models(models_info, X_train_multiclass, X_test_multiclass, y_train_multiclass, y_test_multiclass, classifiers)
