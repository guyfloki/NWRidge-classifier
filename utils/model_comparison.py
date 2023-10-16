import matplotlib.pyplot as plt
import numpy as np

def compare_models(models_info, X_train, X_test, y_train, y_test, classifiers):
    """
    Plots a comparison of training times, accuracies, and decision boundaries for multiple models.
    
    Parameters:
    - models_info: A list of dictionaries, where each dictionary has:
        - 'name': the name of the model.
        - 'train_time': training time for the model.
        - 'accuracy': accuracy score for the model.
    - X_train, X_test, y_train, y_test: Data for visualization.
    - classifiers: Dictionary with classifier instances.
    """
    # Bar plots for training times and accuracies
    names = [model['name'] for model in models_info]
    train_times = [model['train_time'] for model in models_info]
    accuracies = [model['accuracy'] for model in models_info]

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.bar(names, train_times, color=['blue', 'green'])
    plt.title('Training Time Comparison | n-samples=100, centers=5')
    plt.ylabel('Time (seconds)')

    plt.subplot(1, 2, 2)
    plt.bar(names, accuracies, color=['blue', 'green'])
    plt.title('Accuracy Comparison | n-samples=100, centers=5')
    plt.ylim(0, 1.1)
    plt.ylabel('Accuracy')

    plt.tight_layout()
    plt.savefig("model_comparison_5_classes.png")
    plt.show()
    
    # Create a mesh grid for visualization
    X_full = np.vstack([X_train, X_test])
    y_full = np.hstack([y_train, y_test])

    x_min, x_max = X_full[:, 0].min() - 1, X_full[:, 0].max() + 1
    y_min, y_max = X_full[:, 1].min() - 1, X_full[:, 1].max() + 1

    h = .02  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    plt.figure(figsize=(14, 6))

    for i, (name, classifier) in enumerate(classifiers.items()):
        Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        plt.subplot(1, len(classifiers), i + 1)
        plt.contourf(xx, yy, Z, alpha=0.8)
        plt.scatter(X_full[:, 0], X_full[:, 1], c=y_full, edgecolors='k', marker='o')
        plt.title(f'Decision Boundaries with {name}')
    
    plt.tight_layout()
    plt.savefig("decision_boundaries_5_classes.png")
    plt.show()
