import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
import mlflow
import mlflow.sklearn
import json

# -------------------------------
# 1. Load và xử lý dữ liệu MNIST
# -------------------------------
# Tải dữ liệu MNIST (train: 60000 mẫu, test: 10000 mẫu)
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
x, y = mnist.data, mnist.target.astype(int)

# Tách dữ liệu thành tập huấn luyện và kiểm tra
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Chuyển đổi kiểu dữ liệu về float và chuẩn hóa giá trị pixel về [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# ------------------------------------------
# 2. Pipeline cho SVM: Chuẩn hóa, PCA, SVM
# ------------------------------------------
# Bước 2.1: Chuẩn hóa dữ liệu với StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Bước 2.2: Giảm chiều dữ liệu bằng PCA (giữ lại 95% phương sai)
pca = PCA(n_components=0.95)
x_train_pca = pca.fit_transform(x_train_scaled)
x_test_pca = pca.transform(x_test_scaled)

# Lưu scaler và pca để sử dụng cho bước đánh giá sau này
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(pca, 'pca.pkl')

# Bước 2.3: Tối ưu siêu tham số cho SVM bằng GridSearchCV
param_grid_svm = {
    'C': [1, 10],
    'gamma': [0.01, 0.001],
    'kernel': ['rbf']  # Sử dụng kernel RBF
}

grid_svm = GridSearchCV(SVC(), param_grid_svm, cv=3, n_jobs=-1)
grid_svm.fit(x_train_pca, y_train)

# Lấy mô hình SVM với tham số tối ưu
svm_best = grid_svm.best_estimator_
print("Best parameters for SVM:", grid_svm.best_params_)

# Đánh giá mô hình SVM trên tập test
y_pred_svm = svm_best.predict(x_test_pca)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("SVM Accuracy:", accuracy_svm)

# Lưu mô hình SVM đã huấn luyện
joblib.dump(svm_best, 'svm_mnist.pkl')

# ----------------------------------------------------
# 3. Pipeline cho Decision Trees: Huấn luyện trực tiếp
# ----------------------------------------------------
# Tùy chỉnh siêu tham số cho Decision Tree bằng GridSearchCV
param_grid_dt = {
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_dt = GridSearchCV(DecisionTreeClassifier(), param_grid_dt, cv=3, n_jobs=-1)
grid_dt.fit(x_train, y_train)

# Lấy mô hình Decision Tree tốt nhất
dt_best = grid_dt.best_estimator_
print("Best parameters for Decision Tree:", grid_dt.best_params_)

# Đánh giá mô hình Decision Tree trên tập test
y_pred_dt = dt_best.predict(x_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print("Decision Tree Accuracy:", accuracy_dt)

# Lưu mô hình Decision Tree đã huấn luyện
joblib.dump(dt_best, 'decision_tree_mnist.pkl')

# ----------------------------------------------------
# 4. Đánh giá mở rộng và log các chỉ số bằng mlflow
# ----------------------------------------------------
mlflow.set_experiment("MNIST_Classification_Evaluation")

with mlflow.start_run(run_name="Evaluate_SVM_and_DT"):
    # Đánh giá mở rộng cho mô hình SVM
    precision_svm = precision_score(y_test, y_pred_svm, average='weighted')
    recall_svm = recall_score(y_test, y_pred_svm, average='weighted')
    f1_svm = f1_score(y_test, y_pred_svm, average='weighted')
    
    mlflow.log_metric("SVM_accuracy", accuracy_svm)
    mlflow.log_metric("SVM_precision", precision_svm)
    mlflow.log_metric("SVM_recall", recall_svm)
    mlflow.log_metric("SVM_f1", f1_svm)
    
    # Log model SVM
    mlflow.sklearn.log_model(svm_best, "svm_model")
    
    # Lưu classification report của SVM dưới dạng artifact
    report_svm = classification_report(y_test, y_pred_svm, output_dict=True)
    with open("svm_classification_report.json", "w") as f:
        json.dump(report_svm, f, indent=4)
    mlflow.log_artifact("svm_classification_report.json")
    
    # Đánh giá mở rộng cho mô hình Decision Tree
    precision_dt = precision_score(y_test, y_pred_dt, average='weighted')
    recall_dt = recall_score(y_test, y_pred_dt, average='weighted')
    f1_dt = f1_score(y_test, y_pred_dt, average='weighted')
    
    mlflow.log_metric("DT_accuracy", accuracy_dt)
    mlflow.log_metric("DT_precision", precision_dt)
    mlflow.log_metric("DT_recall", recall_dt)
    mlflow.log_metric("DT_f1", f1_dt)
    
    # Log model Decision Tree
    mlflow.sklearn.log_model(dt_best, "decision_tree_model")
    
    # Lưu classification report của Decision Tree dưới dạng artifact
    report_dt = classification_report(y_test, y_pred_dt, output_dict=True)
    with open("dt_classification_report.json", "w") as f:
        json.dump(report_dt, f, indent=4)
    mlflow.log_artifact("dt_classification_report.json")
