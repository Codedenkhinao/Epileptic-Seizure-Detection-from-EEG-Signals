import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib
# Đọc dữ liệu từ file CSV
file_path = r'D:\Epileptic Seizure Detection from EEG Signals\Epileptic Seizure Recognition.csv' 
data = pd.read_csv(file_path)


# Kiểm tra các cột có chứa dữ liệu không phải số
print("\nKiểm tra các cột không phải số:")
print(data.dtypes)

# Chuyển nhãn thành nhị phân (1 cho động kinh, 0 cho không động kinh)
data['y'] = data['y'].apply(lambda x: 1 if x == 1 else 0)

# Kiểm tra lại các nhãn sau khi chuyển đổi
print("\nThông tin về nhãn sau khi chuyển đổi:")
print(data['y'].value_counts())  # In ra số lượng của từng lớp nhãn (0 và 1)

# Tách các đặc trưng (X) và nhãn (y)
X = data.iloc[:, :-1]  # Tất cả các cột trừ cột cuối
y = data['y']  # Cột nhãn

# Kiểm tra xem có cột nào chứa dữ liệu không phải số (chuỗi)
# Nếu có, cần loại bỏ hoặc chuyển đổi chúng
X = X.apply(pd.to_numeric, errors='coerce')  # Chuyển các giá trị không phải số thành NaN
X = X.dropna(axis=1)  # Loại bỏ các cột chứa NaN (nếu có)
# Chia dữ liệu thành tập huấn luyện và kiểm tra
print("\nChia dữ liệu thành tập huấn luyện và kiểm tra...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# In ra kích thước của các tập dữ liệu
print(f"Tập huấn luyện có {X_train.shape[0]} mẫu, tập kiểm tra có {X_test.shape[0]} mẫu.")

# Chuẩn hóa dữ liệu (StandardScaler)
print("\nChuẩn hóa dữ liệu...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(X_train_scaled.shape)  # In ra kích thước của dữ liệu đã chuẩn hóa
print(X_test_scaled.shape)  # In ra kích thước của dữ liệu đã chuẩn hóa
# Khởi tạo mô hình SVM
svm = SVC()

# Định nghĩa các tham số cần tối ưu hóa
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}

# Khởi tạo GridSearchCV để tìm tham số tối ưu
print("\nTìm tham số tối ưu với GridSearchCV...")
grid_search = GridSearchCV(svm, param_grid, cv=5, n_jobs=-1, verbose=3)

# Huấn luyện mô hình với GridSearchCV
grid_search.fit(X_train_scaled, y_train)
# Lưu mô hình đã huấn luyện vào tệp
print("\nLưu mô hình...")
joblib.dump(grid_search, 'svm_model.pkl')  # Lưu mô hình vào tệp svm_model.pkl
# In ra các tham số tối ưu
print("\nCác tham số tối ưu được tìm thấy:")
print(grid_search.best_params_)

# Dự đoán trên tập kiểm tra
print("\nDự đoán trên tập kiểm tra...")
y_pred = grid_search.predict(X_test_scaled)

# Đánh giá mô hình
print("\nĐánh giá mô hình:")
print(classification_report(y_test, y_pred))  # In ra báo cáo phân loại
