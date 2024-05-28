import numpy as np
from scipy.io import loadmat
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

# Load dữ liệu từ tập tin reuters10k.mat
data = loadmat(r'reuters-10k\reuters10k.mat')

# Sử dụng key 'Y' để lấy nhãn
y_true = data['Y'].squeeze()

# Áp dụng K-means clustering
kmeans = KMeans(n_clusters=len(np.unique(y_true)), random_state=42)
y_pred = kmeans.fit_predict(data['X'])

# Đánh giá độ chính xác
# Sử dụng majority voting hoặc phân loại tương ứng để đánh giá độ chính xác của mô hình clustering
def majority_voting(y_true, y_pred):
    labels_mapping = {}
    for cluster in np.unique(y_pred):
        labels_in_cluster = y_true[y_pred == cluster]
        most_common_label = np.bincount(labels_in_cluster).argmax()
        labels_mapping[cluster] = most_common_label
    y_pred_mapped = np.array([labels_mapping[cluster] for cluster in y_pred])
    return y_pred_mapped

y_pred_mapped = majority_voting(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred_mapped)

print("Accuracy:", accuracy)
