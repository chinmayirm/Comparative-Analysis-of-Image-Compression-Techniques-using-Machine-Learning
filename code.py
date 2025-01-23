import cv2
import numpy as np # linear algebra
import pandas as pd 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def calculate_compression(original_file, compressed_file):
    original_size = os.stat(original_file).st_size
    compressed_size = os.stat(compressed_file).st_size
    return original_size / compressed_size

def pca(X, k):
    X = X - np.mean(X, axis=0)
    cov = np.cov(X, rowvar=False)
    eigen_values, eigen_vectors = np.linalg.eigh(cov)
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:,sorted_index]
    eigenvector_subset = sorted_eigenvectors[:, 0:k]
    X_reduced = np.dot(eigenvector_subset.transpose(), X.transpose()).transpose()
      # Reconstruct the compressed image
    X_reconstructed = np.dot(X_reduced, eigenvector_subset.T) + np.mean(X, axis=0)
    return X_reconstructed

def svd(X, k):
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    s = s[:k]
    U = U[:,:k]
    Vt = Vt[:k,:]
    X_reduced = U @ np.diag(s)

     # Reconstruct the compressed image
    X_reconstructed = U @ np.diag(s) @ Vt
    return X_reconstructed

def compress_func(image_path, image_id, image_format):
    # loading the original image
    original = cv2.imread(image_path)

    # lossy compression
    
    cv2.imwrite(f'compressed_lossy_{image_id}.{image_format}', original, [int(cv2.IMWRITE_JPEG_QUALITY), 0])
  
    # PSNR and compression factor for lossy compression
    compressed_lossy = cv2.imread(f'compressed_lossy_{image_id}.{image_format}')
    psnr_lossy = calculate_psnr(original, compressed_lossy)
    compression_lossy = calculate_compression(image_path, f'compressed_lossy_{image_id}.{image_format}')

    # we split the colour channels, apply the function to each channed, then merge everything back
    def apply_to_each_channel(func, img, k):
        channels = cv2.split(img)
        reduced_channels = [func(channel, k) for channel in channels]
        return cv2.merge(reduced_channels)
  compressed_pca = apply_to_each_channel(pca, original, 0)
    compressed_svd = apply_to_each_channel(svd, original, 0)

    # saving the compressed images
    cv2.imwrite(f'compressed_pca_{image_id}.{image_format}', compressed_pca, [int(cv2.IMWRITE_JPEG_QUALITY), 0])
    cv2.imwrite(f'compressed_svd_{image_id}.{image_format}', compressed_svd, [int(cv2.IMWRITE_JPEG_QUALITY), 0])
    print(f"Image {image_id}")
    methods = ['lossy', 'pca', 'svd']
    extensions = [image_format, image_format, image_format]  
    psnrs = []
    compression_factors = []

    for method, extension in zip(methods, extensions):
        compressed = cv2.imread(f'compressed_{method}_{image_id}.{extension}')
        if compressed is None:
            print(f"Unable to load compressed_{method}_{image_id}.{extension}")
            continue
  if original.shape != compressed.shape:
            print(f"Shapes do not match for {method}: original is {original.shape}, compressed is {compressed.shape}")
            continue

        psnrs.append(calculate_psnr(original, compressed))
        compression_factors.append(calculate_compression(image_path, f'compressed_{method}_{image_id}.{extension}'))

    for method, psnr, compression_factor in zip(methods, psnrs, compression_factors):
        print(f"{method.capitalize()} Compression - PSNR: {psnr}, Compression Factor: {compression_factor}")
# calling the function for each image
compress_func('/kaggle/input/finalmathematics-for-machine-learning-assignment-1/part2-image1.jpg', 1, 'jpg')
compress_func('/kaggle/input/finalmathematics-for-machine-learning-assignment-1/part2-image2.jpeg', 2, 'jpeg')

# Define image paths
image_paths = [
    "/kaggle/working/compressed_pca_1.jpg",
    "/kaggle/working/compressed_pca_2.jpeg",
]

# Function to get image size
def get_file_size_mb(file_path):
    size_bytes = os.path.getsize(file_path)
    size_mb = size_bytes / (1024 * 1024)
    return size_mb

sizes = [get_file_size_mb(path) for path in image_paths]

# Write data to CSV file
data = {'Name': ['part2-image1', 'part2-image2'], 'Compressed Image Size[MB]': sizes}

df_submission = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df_submission.to_csv('/kaggle/working/submission.csv', index=False)
