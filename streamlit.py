import streamlit as st
import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
# Tiêu đề ứng dụng
st.title("Phân loại chữ số viết tay MNIST với Streamlit và MLflow")

# ------------------------
# Bước 1: Xử lý dữ liệu
# ------------------------
st.header("1. Xử lý dữ liệu")

# Kiểm tra nếu dữ liệu đã được lưu trong session_state chưa
if "mnist_loaded" not in st.session_state:
    st.write("Đang tải dữ liệu MNIST...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist.data / 255.0, mnist.target.astype(int)
    # Lưu dữ liệu vào session_state để dùng lại
    st.session_state.X = X
    st.session_state.y = y
    st.session_state.mnist_loaded = True
    st.write("Dữ liệu MNIST đã được tải và lưu thành công!")
else:
    st.write("Dữ liệu MNIST đã được tải từ bộ nhớ!")
    X = st.session_state.X
    y = st.session_state.y

# In ra số lượng mẫu của từng label trong dữ liệu gốc
unique_labels, counts = np.unique(y, return_counts=True)
label_counts = {int(k): int(v) for k, v in zip(unique_labels, counts)}
st.write("Số lượng mỗi label trong dữ liệu gốc:", label_counts)

# Hiển thị một vài hình ảnh minh họa và lưu vào session_state
st.subheader("Ví dụ một vài hình ảnh minh họa")
if "example_images" not in st.session_state:
    indices = random.sample(range(len(X)), 5)
    st.session_state.example_images = indices
else:
    indices = st.session_state.example_images

fig, axs = plt.subplots(1, 5, figsize=(12, 3))
for i, idx in enumerate(indices):
    img = X[idx].reshape(28, 28)
    axs[i].imshow(img, cmap='gray')
    axs[i].axis('off')
    axs[i].set_title(f"Label: {y[idx]}")
st.pyplot(fig)

# ------------------------
# Bước 2: Chia tách dữ liệu
# ------------------------
st.header("2. Chia tách dữ liệu")
test_size = st.slider("Chọn tỷ lệ dữ liệu Test", 0.1, 0.5, 0.2, 0.05)
valid_size = st.slider("Chọn tỷ lệ dữ liệu Validation từ Train", 0.1, 0.3, 0.2, 0.05)

if st.button("Chia tách dữ liệu"):
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full, y_train_full, test_size=valid_size, random_state=42, stratify=y_train_full
    )
    st.session_state.X_train = X_train
    st.session_state.X_valid = X_valid
    st.session_state.X_test = X_test
    st.session_state.y_train = y_train
    st.session_state.y_valid = y_valid
    st.session_state.y_test = y_test
    st.session_state.data_split_done = True

# Hiển thị thông tin chia tách nếu đã thực hiện
if st.session_state.get("data_split_done", False):
    st.write(f"Dữ liệu Train: {st.session_state.X_train.shape}")
    st.write(f"Dữ liệu Validation: {st.session_state.X_valid.shape}")
    st.write(f"Dữ liệu Test: {st.session_state.X_test.shape}")

# ------------------------
# Bước 2b: Cân bằng dữ liệu (Oversampling)
# ------------------------
if "X_train" in st.session_state and "y_train" in st.session_state:
    st.subheader("Tăng cường dữ liệu (Oversampling)")
    balance_multiplier = st.slider("Hệ số tăng cường", 1.0, 3.0, 1.0, 0.1, key="balance_multiplier")
    if st.button("Cân bằng dữ liệu"):
        from imblearn.over_sampling import RandomOverSampler
        unique_train, counts_train = np.unique(st.session_state.y_train, return_counts=True)
        max_count = int(max(counts_train))
        target_count = int(max_count * balance_multiplier)
        sampling_strategy = {label: target_count for label in unique_train}
        ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42)
        X_train_bal, y_train_bal = ros.fit_resample(st.session_state.X_train, st.session_state.y_train)
        st.session_state.X_train_bal = X_train_bal
        st.session_state.y_train_bal = y_train_bal
        unique_labels_bal, counts_bal = np.unique(y_train_bal, return_counts=True)
        label_counts_bal = {int(k): int(v) for k, v in zip(unique_labels_bal, counts_bal)}
        st.session_state.balance_info = label_counts_bal

# Hiển thị thông tin cân bằng nếu có
if "balance_info" in st.session_state:
    st.write("Sau khi cân bằng, số lượng mỗi label trong tập huấn luyện:", st.session_state.balance_info)

# ------------------------
# Bước 3: Huấn luyện và đánh giá mô hình
# ------------------------
st.header("3. Huấn luyện và đánh giá mô hình")
model_choice = st.selectbox("Chọn mô hình", ["Decision Tree", "SVM"], key="model_choice")

if st.button("Huấn luyện mô hình"):
    if "X_train" not in st.session_state or "X_valid" not in st.session_state:
        st.error("Bạn cần chia tách dữ liệu trước!")
        st.stop()
    # Sử dụng dữ liệu đã cân bằng nếu có, ngược lại dùng dữ liệu gốc
    if "X_train_bal" in st.session_state and "y_train_bal" in st.session_state:
        X_train_used = st.session_state.X_train_bal
        y_train_used = st.session_state.y_train_bal
    else:
        X_train_used = st.session_state.X_train
        y_train_used = st.session_state.y_train
    X_valid = st.session_state.X_valid
    y_valid = st.session_state.y_valid

    with mlflow.start_run():
        if model_choice == "Decision Tree":
            st.session_state.model = DecisionTreeClassifier(max_depth=20, random_state=42)
        else:
            scaler = StandardScaler()
            X_train_used_scaled = scaler.fit_transform(X_train_used)
            X_valid_scaled = scaler.transform(X_valid)
            st.session_state.scaler = scaler
            st.session_state.model = SVC(kernel='rbf', C=10, gamma=0.01, random_state=42)
        if model_choice == "SVM":
            st.session_state.model.fit(X_train_used_scaled, y_train_used)
            y_pred = st.session_state.model.predict(X_valid_scaled)
        else:
            st.session_state.model.fit(X_train_used, y_train_used)
            y_pred = st.session_state.model.predict(X_valid)
        st.session_state.trained_model_name = model_choice
        st.session_state.train_accuracy = accuracy_score(y_valid, y_pred)
        st.session_state.train_report = classification_report(y_valid, y_pred)
        mlflow.log_param("model", model_choice)
        mlflow.log_metric("accuracy", st.session_state.train_accuracy)
        mlflow.sklearn.log_model(st.session_state.model, "model")

# Hiển thị kết quả huấn luyện nếu có
if "train_accuracy" in st.session_state:
    st.write(f"Độ chính xác trên tập validation: {st.session_state.train_accuracy:.4f}")
if "train_report" in st.session_state:
    st.text(st.session_state.train_report)

# ------------------------
# Bước 4: Demo dự đoán
# ------------------------
st.header("4. Demo dự đoán")
uploaded_file = st.file_uploader("Tải lên hình ảnh chữ số viết tay", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    from PIL import Image
    if st.session_state.get("model") is None:
        st.error("Bạn cần huấn luyện mô hình trước khi dự đoán!")
    else:
        if st.button("Dự đoán từ ảnh tải lên"):
            image = Image.open(uploaded_file).convert('L')
            image = image.resize((28, 28))
            img_array = np.array(image).reshape(1, -1) / 255.0
            if st.session_state.trained_model_name == "SVM" and st.session_state.get("scaler") is not None:
                img_array = st.session_state.scaler.transform(img_array)
            prediction = st.session_state.model.predict(img_array)
            st.image(image, caption="Hình ảnh tải lên", use_container_width=True)
            st.write(f"Dự đoán: {prediction[0]}")

st.header("5. Tracking MLflow")

try:
    import tempfile
    import shutil
    from mlflow.tracking import MlflowClient
    client = MlflowClient()

    # Lấy danh sách thí nghiệm từ MLflow
    experiments = mlflow.search_experiments()

    if experiments:
        st.write("#### Danh sách thí nghiệm")
        experiment_data = []
        for exp in experiments:
            experiment_data.append({
                "Experiment ID": exp.experiment_id,
                "Experiment Name": exp.name,
                "Artifact Location": exp.artifact_location
            })
        st.dataframe(pd.DataFrame(experiment_data))

        # Chọn thí nghiệm để xem chi tiết
        selected_exp_id = st.selectbox(
            "🔍 Chọn thí nghiệm để xem chi tiết",
            options=[exp.experiment_id for exp in experiments]
        )

        # Lấy danh sách runs trong thí nghiệm đã chọn
        runs = mlflow.search_runs(selected_exp_id)
        if not runs.empty:
            st.write("#### Danh sách runs")
            st.dataframe(runs)

            # Chọn run để xem chi tiết
            selected_run_id = st.selectbox(
                "🔍 Chọn run để xem chi tiết",
                options=runs["run_id"]
            )

            # Hiển thị chi tiết run
            run = mlflow.get_run(selected_run_id)
            st.write("##### Thông tin run")
            st.write(f"*Run ID:* {run.info.run_id}")
            st.write(f"*Experiment ID:* {run.info.experiment_id}")
            st.write(f"*Start Time:* {run.info.start_time}")

            # Hiển thị metrics
            st.write("##### Metrics")
            st.json(run.data.metrics)

            # Hiển thị params
            st.write("##### Params")
            st.json(run.data.params)

            # Hiển thị artifacts và cho phép tải xuống hoặc trực quan hóa
            artifacts = client.list_artifacts(run.info.run_id)
            if artifacts:
                st.write("##### Artifacts")
                for artifact in artifacts:
                    try:
                        with tempfile.TemporaryDirectory() as tmp_dir:
                            artifact_path = client.download_artifacts(run.info.run_id, artifact.path, dst_path=tmp_dir)
                            local_artifact_path = shutil.copy(artifact_path, tmp_dir)
                            st.write(f"- {artifact.path}")
                            if artifact.path.endswith((".png", ".jpg", ".jpeg")):
                                st.image(local_artifact_path, caption=artifact.path)
                            elif artifact.path.endswith((".csv", ".txt")):
                                with open(local_artifact_path, "r", encoding='utf-8') as file:
                                    st.text(file.read())
                            with open(local_artifact_path, "rb") as file:
                                st.download_button(
                                    label=f"📥 Tải xuống {artifact.path}",
                                    data=file.read(),
                                    file_name=artifact.path
                                )
                    except PermissionError:
                        st.error(f"Không thể tải artifact do lỗi quyền truy cập: {artifact.path}")
            else:
                st.warning("Không có artifact nào trong run này.")
        else:
            st.warning("Không có runs nào trong thí nghiệm này.")
    else:
        st.warning("Không có thí nghiệm nào được tìm thấy.")
except Exception as e:
    st.error(f"Đã xảy ra lỗi khi lấy danh sách thí nghiệm: {e}")
