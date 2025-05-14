import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

data=pd.read_csv (r"D:\Epileptic Seizure Detection from EEG Signals\Epileptic Seizure Recognition.csv")
pd.set_option('display.max_columns',None)
data.head()
data.shape
data.info()
data.describe()
data.isna().sum()
data.drop(columns=['Unnamed'],inplace=True)

sns.set_theme(style="whitegrid",context="talk")

unique_classes=data['y'].unique()

fig,axes=plt.subplots(nrows=len(unique_classes),ncols=1,figsize=(9,4*len(unique_classes)))
fig.suptitle('EEG signals by CLass (one sample per class)',fontsize=20,fontweight='bold',y=1.02)

for i,class_label in enumerate(unique_classes):
    class_data=data[data['y']==class_label]
    random_sample=class_data.sample(n=1).drop(columns='y') #Exclude the 'y' column
    
    signal_data = random_sample.melt(var_name='Time', value_name='Signal Value')
    sns.lineplot(
        data=signal_data,
        x='Time',
        y='Signal Value',
        ax=axes[i],
        color=sns.color_palette("husl", len(unique_classes))[i],  # Distinct color for each class
    )

    axes[i].set_title(f'Class {class_label}', fontsize=16, fontweight='bold')
    axes[i].set_ylabel('Signal Value', fontsize=12)
    axes[i].set_xlabel('Time', fontsize=12)
    axes[i].tick_params(axis='both', which='major', labelsize=10)

plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the title
sns.despine()

plt.savefig("my_plot1.png")

data['y'].value_counts()
data.loc[data['y'] != 1, 'y'] = 0
sns.set_theme(style="whitegrid", context="talk")

# Create the count plot
plt.figure(figsize=(9, 5))
ax = sns.countplot(x=data['y'], palette="viridis")

ax.set_title("Class Distribution", fontsize=10, fontweight='bold')
ax.set_xlabel("Classes", fontsize=14)
ax.set_ylabel("Count", fontsize=14)
ax.tick_params(axis='both', labelsize=12)

# Add count annotations on the bars
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}",
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center',
                fontsize=12, color='black',
                xytext=(0, 5),
                textcoords='offset points')

plt.tight_layout()
sns.despine()

plt.savefig("my_plot2.png")

X = data.drop("y",axis=1)
y = data['y']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,shuffle=True)


X_train = X_train.values.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.values.reshape(X_test.shape[0], 1, X_test.shape[1])
X_train.shape

model = Sequential([
    LSTM(128, activation='relu', return_sequences=True, input_shape=(1, 178)),
    LSTM(128, activation='relu', return_sequences=True),
    LSTM(64, activation='relu', return_sequences=True),
    LSTM(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

history = model.fit(X_train, y_train, epochs=25, batch_size=65, validation_data=(X_test,y_test))

import matplotlib.pyplot as plt

# Convert accuracy to percentage and plot
plt.figure(figsize=(8, 6))

# Plot training and validation accuracy as percentages
plt.plot([x * 100 for x in history.history['accuracy']], label='Training Accuracy')
plt.plot([x * 100 for x in history.history['val_accuracy']], label='Validation/Test Accuracy')

# Set title and labels
plt.title('Training and Validation Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')

# Set y-axis limits from 0 to 100
plt.ylim(0, 100)

# Add legend
plt.legend()

# Display the plot
plt.savefig("my_plot3.png")

y_pred_prob2 = model.predict(X_test)  # Get probabilities
y_pred = np.argmax(y_pred_prob2, axis=1)  # Convert to class labels
y_true = np.argmax(y_test, axis=1)  # Convert one-hot-encoded true labels to class labels

cm = confusion_matrix(y_true, y_pred)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Class 0', 'Class 1'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# Generate and print the classification report
report = classification_report(y_true, y_pred, target_names=['Class 0', 'Class 1'])
print(report)
model.save("EEG_LSTM_98.h5")