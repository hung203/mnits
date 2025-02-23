import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import titanic.processing_titanic as processing_titanic

X_train, X_temp, y_train, y_temp, X_valid, X_test, y_valid, y_test=processing_titanic.process()
# Bắt đầu một run mới trong MLFlow
ex=mlflow.set_experiment(experiment_name='experiment1')
with mlflow.start_run(experiment_id=ex.experiment_id):
    # Khởi tạo mô hình Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Huấn luyện mô hình với Cross Validation trên tập training và validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    mean_cv_accuracy = np.mean(cv_scores)
    
    # Log các thông số và metrics
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("mean_cv_accuracy", mean_cv_accuracy)
    
    # Huấn luyện mô hình trên toàn bộ tập training
    model.fit(X_train, y_train)
    
    # Dự đoán trên tập validation
    y_valid_pred = model.predict(X_valid)
    
    # Tính toán các metrics trên tập validation
    valid_accuracy = accuracy_score(y_valid, y_valid_pred)
    valid_precision = precision_score(y_valid, y_valid_pred)
    valid_recall = recall_score(y_valid, y_valid_pred)
    valid_f1 = f1_score(y_valid, y_valid_pred)
    
    # Log các metrics trên tập validation
    mlflow.log_metric("valid_accuracy", valid_accuracy)
    mlflow.log_metric("valid_precision", valid_precision)
    mlflow.log_metric("valid_recall", valid_recall)
    mlflow.log_metric("valid_f1", valid_f1)
    
    # Dự đoán trên tập test
    y_test_pred = model.predict(X_test)
    
    # Tính toán các metrics trên tập test
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    
    # Log các metrics trên tập test
    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.log_metric("test_precision", test_precision)
    mlflow.log_metric("test_recall", test_recall)
    mlflow.log_metric("test_f1", test_f1)
    
    # Log mô hình
    mlflow.sklearn.log_model(model, "random_forest_model")
    # Sau khi train và log model
    mlflow.sklearn.save_model(model, "model/")
    # In kết quả
    print(f"Mean CV Accuracy: {mean_cv_accuracy}")
    print(f"Validation Accuracy: {valid_accuracy}")
    print(f"Test Accuracy: {test_accuracy}")