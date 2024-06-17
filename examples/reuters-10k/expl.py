import scipy.io as sio

# Đường dẫn đến file .mat của bạn
mat_file = r'examples\reuters-10k\reuters10k.mat'

# Đọc file .mat
mat_contents = sio.loadmat(mat_file)

# Hiển thị các tên biến trong file .mat
print(mat_contents.keys())

# Giả sử file .mat chứa các biến 'X' và 'Y', ta có thể truy cập và hiển thị dữ liệu như sau:
X = mat_contents['X']
Y = mat_contents['Y']

print('Shape of X:', X.shape)
print('Shape of Y:', Y.shape)

# Hiển thị một phần dữ liệu để kiểm tra
print('X:', X[:5])
print('Y:', Y[:5])
