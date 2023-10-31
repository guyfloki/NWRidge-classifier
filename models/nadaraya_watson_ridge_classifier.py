
import numpy as np
from typing import Callable, Union
from scipy.sparse import issparse, csr_matrix
from concurrent.futures import ThreadPoolExecutor

class AdamWOptimizer:
    """Class for AdamW optimizer.

    Attributes:
        lr (float): Learning rate.
        beta1 (float): Exponential decay rate for the first moment estimates.
        beta2 (float): Exponential decay rate for the second moment estimates.
        epsilon (float): Small constant for numerical stability.
        weight_decay (float): Weight decay coefficient.
        max_iter (int): Maximum number of iterations for optimization.
    """
    def __init__(self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, 
                 epsilon: float = 1e-8, weight_decay: float = 0.01, max_iter: int = 1):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.max_iter = max_iter

    def optimize(self, loss_func: Callable[[np.ndarray], float], initial_params: np.ndarray) -> np.ndarray:
        """Optimize the given loss function.

        Args:
            loss_func (Callable): The loss function to optimize.
            initial_params (np.ndarray): Initial parameters to start optimization.

        Returns:
            np.ndarray: Optimized parameters.
        """
        m = np.zeros_like(initial_params)
        v = np.zeros_like(initial_params)
        t = 0
        params = initial_params.copy()
        for _ in range(self.max_iter):
            grad = loss_func(params)
            t += 1
            m = self.beta1 * m + (1 - self.beta1) * grad
            v = self.beta2 * v + (1 - self.beta2) * grad ** 2
            m_hat = m / (1 - self.beta1 ** t)
            v_hat = v / (1 - self.beta2 ** t)
            params -= self.lr * (m_hat / (np.sqrt(v_hat) + self.epsilon) + self.weight_decay * params)
        return params

class NadarayaWatsonRidgeClassifier:
    """Nadaraya-Watson Ridge Classifier.

    This classifier combines the Nadaraya-Watson kernel regression technique
    with a ridge regularization for multi-class classification.

    Attributes:
        alpha (float): Regularization strength for ridge regression.
        h (float): Bandwidth parameter for the kernel function.
        kernel_func (Callable): Kernel function.
    """
    def __init__(self, alpha: float = 1.0, h: float = 0.5, 
             kernel_func: Callable = None, batch_size: int = 1000):
        self.alpha = alpha
        self.h = h
        self.kernel_func = kernel_func if kernel_func else self.gaussian_kernel
        self.w = None
        self.b = None
        self.X_train = None
        self.y_train = None
        self.kernel_cache = {}
        self.n_classes = None
        self.batch_size = batch_size

    @staticmethod
    def gaussian_kernel(X1: np.ndarray, X2: np.ndarray, h: float) -> np.ndarray:
        """Compute Gaussian Kernel.

        Args:
            X1 (np.ndarray): First data matrix.
            X2 (np.ndarray): Second data matrix.
            h (float): Bandwidth parameter.

        Returns:
            np.ndarray: Computed Gaussian Kernel.
        """
        pairwise_diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
        squared_diff = np.square(pairwise_diff)
        K = -0.5 * np.sum(squared_diff, axis=-1)
        return np.exp(K / h ** 2)

    def fit(self, X: Union[np.ndarray, csr_matrix], y: np.ndarray) -> None:
        """Train the model.

        Args:
            X (Union[np.ndarray, csr_matrix]): Feature matrix.
            y (np.ndarray): Label vector.

        Returns:
            None
        """
        self._clear_kernel_cache()
        if issparse(X):
            X = csr_matrix(X)
        self.X_train = X
        self.n_classes = len(np.unique(y))
        y_onehot = np.zeros((y.size, self.n_classes))
        y_onehot[np.arange(y.size), y] = 1
        self.y_train = y_onehot
        n_features = X.shape[1]
        initial_params = np.zeros((n_features + 1) * self.n_classes)

        def loss(params: np.ndarray) -> float:
            w = params[:-self.n_classes].reshape((n_features, self.n_classes))
            b = params[-self.n_classes:]
            linear_preds = X.dot(w) + b
            kernel_corrections = self._nadaraya_watson(X, y_onehot - linear_preds, X)
            logits = linear_preds + kernel_corrections
            return np.mean(self.softmax_cross_entropy_loss(y_onehot, logits)) + self.alpha * np.sum(w ** 2)

        optimizer = AdamWOptimizer()
        optimal_params = optimizer.optimize(loss, initial_params)
        self.w = optimal_params[:-self.n_classes].reshape((n_features, self.n_classes))
        self.b = optimal_params[-self.n_classes:]

    def _clear_kernel_cache(self) -> None:
        """Clear the kernel cache.

        Returns:
            None
        """
        self.kernel_cache.clear()



    def _nadaraya_watson(self, X: np.ndarray, residuals: np.ndarray, 
                         X_query: np.ndarray) -> np.ndarray:
        """Compute Nadaraya-Watson estimates.

        Args:
            X (np.ndarray): Data matrix.
            residuals (np.ndarray): Residuals for each data point.
            X_query (np.ndarray): Query points for estimation.
            batch_size (int, optional): Batch size for computation. Defaults to 50.

        Returns:
            np.ndarray: Nadaraya-Watson estimates for query points.
        """
        n_batches = (X.shape[0] + self.batch_size - 1) // self.batch_size  
        kernel_corrections = np.zeros((X_query.shape[0], self.n_classes))

        def process_batch(i: int) -> np.ndarray:
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, X.shape[0])
            X_chunk = X[start_idx:end_idx]
            residuals_chunk = residuals[start_idx:end_idx]
            weights = self._compute_kernel_chunk(X_chunk, X_query)
            return np.dot(weights.T, residuals_chunk)

        with ThreadPoolExecutor() as executor:
            batch_results = list(executor.map(process_batch, range(n_batches)))

        for weighted_sums_batch in batch_results:
            kernel_corrections += weighted_sums_batch

        return kernel_corrections
    
    def _compute_kernel_chunk(self, X_chunk: np.ndarray, 
                              X_query: np.ndarray) -> np.ndarray:
        """Compute a chunk of kernel matrix.

        Args:
            X_chunk (np.ndarray): Subset of data points.
            X_query (np.ndarray): Query points for computation.

        Returns:
            np.ndarray: Computed kernel values for the chunk.
        """
        key = (tuple(X_chunk.flatten()), tuple(X_query.flatten()))
        if key in self.kernel_cache:
            return self.kernel_cache[key]

        kernel_values = self.kernel_func(X_chunk, X_query, self.h)
        self.kernel_cache[key] = kernel_values
        return kernel_values

    @staticmethod
    def softmax(logits: np.ndarray) -> np.ndarray:
        """Compute softmax of logits.

        Args:
            logits (np.ndarray): Input logits.

        Returns:
            np.ndarray: Softmax probabilities.
        """
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    @staticmethod
    def softmax_cross_entropy_loss(y_true: np.ndarray, logits: np.ndarray) -> np.ndarray:
        """Compute softmax cross entropy loss.

        Args:
            y_true (np.ndarray): Ground truth labels in one-hot encoding.
            logits (np.ndarray): Predicted logits.

        Returns:
            np.ndarray: Softmax cross entropy loss.
        """
        probabilities = NadarayaWatsonRidgeClassifier.softmax(logits)
        return -np.sum(y_true * np.log(probabilities + 1e-10), axis=-1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for given data points.

        Args:
            X (np.ndarray): Data points for prediction.

        Returns:
            np.ndarray: Predicted class labels.
        """
        linear_preds = X.dot(self.w) + self.b
        kernel_corrections = self._nadaraya_watson(self.X_train, self.y_train - self.X_train.dot(self.w), X)
        corrected_logits = linear_preds + kernel_corrections
        return np.argmax(self.softmax(corrected_logits), axis=1)
    
