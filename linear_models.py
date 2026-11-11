import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationResult,
    RegressionInput,
    RegressionResult,
    ClassificationMetrics
)

def train_linear_regression(data: RegressionInput) -> RegressionResult:
    """Обучите собственную линейную регрессию на train и верните прогноз для test."""
    X_train = np.array(data.train_features, dtype=np.float64)
    y_train = np.array(data.train_target, dtype=np.float64)
    X_test = np.array(data.test_features, dtype=np.float64)

    # Добавляем столбец единиц для bias
    X_train_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]
    X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]

    # Инициализация весов
    n_features = X_train_b.shape[1]
    weights = np.zeros(n_features)
    
    lr = data.learning_rate
    epochs = data.epochs
    m = X_train.shape[0]
    
    loss_history = []

    for epoch in range(epochs):
        # Предсказание
        y_pred = X_train_b.dot(weights)
        
        # Ошибка
        error = y_pred - y_train
        
        # Градиент (MSE)
        gradient = (2/m) * X_train_b.T.dot(error)
        
        # Обновление весов
        weights -= lr * gradient
        
        # Сохранение истории потерь
        mse = np.mean(error ** 2)
        loss_history.append(float(mse))

    # Предсказание на тесте
    test_preds = X_test_b.dot(weights)

    return RegressionResult(
        weights=weights[1:],  # исключаем bias
        bias=float(weights[0]),
        loss_history=loss_history,
        predictions=test_preds
    )


def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    """Обучите собственный логистический классификатор на train и верните прогноз для test."""
    X_train = np.array(data.train_features, dtype=np.float64)
    y_train = np.array(data.train_target, dtype=np.float64)
    X_test = np.array(data.test_features, dtype=np.float64)

    # Добавляем столбец единиц для bias
    X_train_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]
    X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]

    # Инициализация весов
    n_features = X_train_b.shape[1]
    weights = np.zeros(n_features)
    
    lr = data.learning_rate
    epochs = data.epochs
    m = X_train.shape[0]
    
    loss_history = []

    def sigmoid(z):
        # Защита от переполнения
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    for epoch in range(epochs):
        # Линейная комбинация
        z = X_train_b.dot(weights)
        # Вероятности
        y_pred = sigmoid(z)
        
        # Градиент бинарной кросс-энтропии
        gradient = (1/m) * X_train_b.T.dot(y_pred - y_train)
        
        # Обновление весов
        weights -= lr * gradient
        
        # Вычисление функции потерь
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(y_train * np.log(y_pred) + (1 - y_train) * np.log(1 - y_pred))
        loss_history.append(float(loss))

    # Предсказание на тесте
    test_probs = sigmoid(X_test_b.dot(weights))
    test_preds = (test_probs >= 0.5).astype(int)

    return ClassificationResult(
        weights=weights[1:],
        bias=float(weights[0]),
        loss_history=loss_history,
        probabilities=test_probs,
        predictions=test_preds
    )


def _get_metrics(y_true, y_pred, y_prob):
    return ClassificationMetrics(
        roc_auc=float(roc_auc_score(y_true, y_prob)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0))
    )


def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    """Сравните готовые Logistic Regression, Decision Tree и KNN на одном split."""
    X_train = np.array(data.train_features, dtype=np.float64)
    y_train = np.array(data.train_target, dtype=np.float64)
    X_test = np.array(data.test_features, dtype=np.float64)
    y_test = np.array(data.test_target, dtype=np.float64)

    # 1. Своя логистическая регрессия (только что обученная)
    own_result = train_logistic_classifier(data)
    own_metrics = _get_metrics(y_test, own_result.predictions, own_result.probabilities)

    # 2. Sklearn Logistic Regression
    sk_log_reg = SklearnLogisticRegression(max_iter=1000)
    sk_log_reg.fit(X_train, y_train)
    sk_preds = sk_log_reg.predict(X_test)
    sk_probs = sk_log_reg.predict_proba(X_test)[:, 1]
    sk_metrics = _get_metrics(y_test, sk_preds, sk_probs)

    # 3. Decision Tree
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    dt_preds = dt.predict(X_test)
    dt_probs = dt.predict_proba(X_test)[:, 1]
    dt_metrics = _get_metrics(y_test, dt_preds, dt_probs)

    # 4. KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    knn_preds = knn.predict(X_test)
    knn_probs = knn.predict_proba(X_test)[:, 1]
    knn_metrics = _get_metrics(y_test, knn_preds, knn_probs)

    return BaselineComparisonResult(
        logistic_regression=sk_metrics,
        decision_tree=dt_metrics,
        knn=knn_metrics
    )
