#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Libs
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler



# # Data Setup

# In[2]:


df = pd.read_csv('/kaggle/input/ecg-dataset/ecg.csv',header=None)
df.head(10)


# In[3]:


#Adding Label
adding_columns = [f'f{i}' for i in range(0, 188)] + ['label']
df.columns = adding_columns


# In[4]:


df


# # Exploratory Data Analysis (EDA)

# In[5]:


#EDA Data Analysis
df.describe()


# In[6]:


df.shape


# In[7]:


#Data Distribution
print(df['label'].value_counts(normalize=True) * 100)


# In[8]:


#Missing Value (zeroed value)
print(df.isnull().sum())


# # Outlier Removal

# In[9]:


#Outlier check
z_scores = zscore(df.drop('label', axis=1))

#Check outliers where Z_score is more than 3
outliers = (abs(z_scores) > 3)

#Count outliers per column
outlier_counts = outliers.sum(axis=0)
print(outlier_counts)


# In[10]:


#Check total outlier, since number of outlieris less than 10%, just delete the outlier
num_outlier_rows = outliers.any(axis=1).sum()
num_outlier_rows


# In[11]:


#Z-score calculation except rom label column
z_scores = zscore(df.drop('label', axis=1))

#Outlier rows detection
mask = (abs(z_scores) <= 3).all(axis=1)

#save rows that doesn't contain outliers
df_clean = df[mask].reset_index(drop=True)

print(f"Initial Data: {df.shape[0]} rows")
print(f"Cleaned Outlier Data: {df_clean.shape[0]} rows")


# In[12]:


df_clean


# In[13]:


#save the cleaned dataset
df_clean.to_csv('cleaned_ECG_data.csv', index=False)
df = df_clean


# # Check Abnormal Data Diversity With K-Means

# In[14]:


X1 = df.iloc[:, :-1].values
y1 = df.iloc[:, -1].values 

#normalization (may be deleted)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X1)

n_clusters = 7
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

plt.figure(figsize=(15, 10))
for cluster_id in range(n_clusters):
    rows = (n_clusters + 4) // 5
    plt.subplot(rows, 5, cluster_id + 1)
    cluster_indices = np.where(clusters == cluster_id)[0]
    
    for idx in cluster_indices:
        color = 'red' if y1[idx] == 1 else 'blue'
        plt.plot(X1[idx], color=color, alpha=0.3)

    plt.title(f'Cluster {cluster_id}')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.grid(True)

plt.suptitle('ECG Beat Clusters Colored by Label (Red=1, Blue=0)', fontsize=16)
plt.tight_layout()
plt.show()


# # PCA Dimension Reduction

# In[15]:


#PCA dimension reduction
print(df.columns.tolist())
X = df.drop('label', axis=1)
y = df['label']

#Feature Normalization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Implement PCA to 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)


# In[16]:


#PCA result
plt.figure(figsize=(8,6))
custom_palette = {0: 'blue', 1: 'red'}
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette=custom_palette)
y_labels = {0: 'normal', 1: 'abnormal'}
plt.legend(title='Condition', labels=['normal', 'abnormal'])
plt.title("2D PCA Visualization")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()


# # Data Preparation

# In[17]:


#Dataset Preparation
X = df.drop('label', axis=1).values
y = df['label'].values


# In[18]:


#1 dimension only since ECG data is 1 dimensional (Voltage against time)
X = X.reshape((X.shape[0], X.shape[1], 1))


# In[19]:


#Encode the labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)


# In[20]:


#Spiting DS
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)


# # Model Definition

# In[21]:


#1Conv Train
model = Sequential([
    Conv1D(64, kernel_size=5, activation='relu', input_shape=(X.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),
    
    Conv1D(128, kernel_size=5, activation='relu'),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(y_categorical.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


# In[22]:


#save model visual to output
from tensorflow.keras.utils import plot_model
plot_model(model, to_file='/kaggle/working/model.png', show_shapes=True, show_layer_names=True)


# # Model Training

# In[23]:


#checkpoint to get best model only
checkpoint = ModelCheckpoint(
    '/kaggle/working/ecg_best_model.h5',
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)
#train
model.fit(
    X_train, y_train,
    epochs=500,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[checkpoint]
)


# # Evaluation

# In[24]:


#Accuracy
loss, acc = model.evaluate(X_test, y_test)
print(f"Akurasi: {acc:.4f}")


# In[25]:


#Confusion Matrix
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true_classes, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()


# # Prediction

# In[26]:


#Prediction
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)


# In[27]:


#Prediction confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_true_classes, y_pred_classes)
ConfusionMatrixDisplay(cm, display_labels=le.classes_).plot()


# # Final Report

# In[28]:


#Clarification Report
from sklearn.metrics import classification_report
target_names = [str(label) for label in le.classes_]
print(classification_report(y_true_classes, y_pred_classes, target_names=target_names))


# In[29]:


#ROC-AUC
from sklearn.metrics import roc_auc_score
roc_auc = roc_auc_score(y_test, y_pred, multi_class='ovr')
print(f"ROC-AUC Score: {roc_auc:.4f}")


# # Prediction

# In[30]:


#Data for Testing
data = [
    952, 954, 956, 955, 955, 953, 952, 952, 951, 955, 953, 954, 952, 953, 952, 955,
    957, 958, 958, 962, 963, 964, 963, 965, 963, 967, 969, 971, 973, 973, 972, 971,
    973, 973, 972, 968, 966, 968, 970, 973, 969, 966, 960, 964, 965, 970, 971, 970,
    967, 964, 962, 961, 960, 954, 950, 951, 951, 951, 950, 950, 950, 948, 952, 949,
    949, 944, 940, 943, 944, 947, 948, 944, 941, 943, 945, 945, 945, 942, 938, 936,
    933, 926, 927, 920, 910, 909, 919, 940, 963, 986, 1020, 1068, 1121, 1167, 1193,
    1201, 1182, 1136, 1069, 1010, 967, 939, 924, 914, 917, 924, 931, 934, 934, 937,
    940, 942, 942, 941, 940, 939, 940, 938, 938, 936, 934, 934, 938, 938, 938, 935,
    935, 936, 939, 938, 938, 939, 935, 935, 938, 937, 937, 935, 935, 937, 938, 939,
    938, 938, 936, 940, 938, 939, 938, 936, 936, 938, 938, 941, 939, 938, 933, 935,
    935, 938, 938, 936, 936, 937, 938, 941, 941, 939, 939, 940, 943, 943, 941, 941,
    939, 938, 943, 943, 943, 943, 938, 941, 942, 941, 938, 935, 931, 932
]
data_category = 0


# In[31]:


#Predict Function
def predict(signal):
    signal = np.array(data)
    # signal = (signal - np.mean(signal)) / np.std(signal) #Input normalization
    signal = signal.reshape(1, -1, 1)
    y_pred = model.predict(signal)
    predicted_class = np.argmax(y_pred)
    predicted_label = le.inverse_transform([predicted_class])[0]
    print(f"Prediction Result: {predicted_label}, expected to be: {data_category}")
    


# In[32]:


predict(data)


# # Save Model

# In[33]:


model.save('/kaggle/working/ecg_last_model.h5')

