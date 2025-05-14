import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# Đọc dữ liệu
file_path = r'D:\Epileptic Seizure Detection from EEG Signals\Epileptic Seizure Recognition.csv'
data = pd.read_csv(file_path)

# Chuyển nhãn thành nhị phân
data['y'] = data['y'].apply(lambda x: 1 if x == 1 else 0)

# Tách đặc trưng và nhãn
X = data.iloc[:, 1:-1].values  # Bỏ cột đầu (id) và cột cuối (y)
y = data['y'].values

# Chuyển đổi dữ liệu không hợp lệ thành NaN rồi thay bằng giá trị trung bình
X = pd.DataFrame(X)
X = X.apply(pd.to_numeric, errors='coerce')
X = X.fillna(X.mean())

# Chia tập train/test
X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Định dạng lại cho CNN (reshape về 2D + 1 channel)
X_train_scaled = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1, 1)
X_test_scaled = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1, 1)

# Xây dựng mô hình CNN
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 1), activation='relu', input_shape=(X_train_scaled.shape[1], 1, 1)))
model.add(MaxPooling2D(pool_size=(2, 1)))
model.add(Conv2D(64, kernel_size=(3, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 1)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile mô hình
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Callback lưu mô hình tốt nhất
checkpoint_path = r'D:\Workspace\Deep_Learning\EEG\cnn_best_model.h5'
checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor='val_accuracy', save_best_only=True, verbose=1)

# Huấn luyện mô hình
print("\n Đang huấn luyện mô hình CNN...")
history = model.fit(
    X_train_scaled, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test_scaled, y_test),
    callbacks=[checkpoint],
    verbose=1
)

# Dự đoán và đánh giá
print("\n Dự đoán và đánh giá mô hình...")
y_pred = (model.predict(X_test_scaled) > 0.5).astype("int32")
print("\n Classification Report:")
print(classification_report(y_test, y_pred))

# Đồ thị Loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss during training')
plt.legend()
plt.savefig(r"D:\Workspace\Deep_Learning\EEG\plot1.png")

# Đồ thị Accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy during training')
plt.legend()
plt.savefig(r"D:\Workspace\Deep_Learning\EEG\plot2.png")

print(f"\n Mô hình tốt nhất đã được lưu tại: {checkpoint_path}")
