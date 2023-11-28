# -*- coding: utf-8 -*-
"""
CW 1 of CS485
"""
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import seaborn as sns
import itertools
from matplotlib import pyplot as plt

import scipy.io
mat = scipy.io.loadmat('face.mat')
data = mat['X'].reshape(2576, 52, 10)
Y = mat['l'].reshape(52, 10)



train, test = data[:, :, :8], data[:, :, 8:]  # 8:2 Split.
# 1st dim is each image, 2nd dim is each person.

train_flat, test_flat = train.reshape(2576, 52*8).T, test.reshape(2576, 52*2).T
train_y, test_y = Y[:, :8].flatten(), Y[:, 8:].flatten()

# Visualize the images
def show_arr_imgs(arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(arr[i].reshape(46, 56).T, cmap='gray')
        ax.set_title(f'Image {i+1}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    
def show_graph(data):
    x_values = np.arange(len(data))
    
    plt.figure(figsize=(8, 8))  # Adjust the figure size if necessary
    plt.plot(x_values, data, marker='.', color='b', linewidth=1, markersize=0)
    
    plt.ylabel('Eigenvalue')
    
    plt.tight_layout()
    plt.show()

# def recons_ori(inp_norm, eigen_vec, top_m):
#     out = 0
#     for i in range(0, top_m):
#         weight = np.dot(inp_norm, eigen_vec[:,i])
#         out += eigen_vec[:,i] * weight
#     return out

# # Visualize the images, for match and failed cases.
# fig, axes = plt.subplots(2, 8, figsize=(20, 6))
# for i, ax in enumerate(axes.flat):
#     ax.imshow(test_flat[i].reshape(46, 56).T, cmap='gray')
#     ax.set_title(f'Img {i}| gt: {test_y[i]} pred: {test_pred[i]}')
#     ax.axis('off')
# plt.tight_layout()
# plt.show()


def recons(inp_norm, eigen_vec, top_m):
    weights = np.dot(inp_norm, eigen_vec[:, :top_m])
    out = np.dot(weights, eigen_vec[:, :top_m].T)
    return out

def show_one(img, text='Sample'): # Plot a single image
    plt.imshow(img.reshape(46, 56).T, cmap='gray')
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.title(text)  # Optional: add a title
    plt.show()
    
def show_conf_mat(confu_mat):
    # Create a heatmap
    plt.figure(figsize=(10, 10)) # dont delete this!
    ax = sns.heatmap(confu_mat, annot=False, cmap='hot', square=True)
    ax.set_yticks(np.arange(0, len(ax.get_xticklabels())*2, 2))

    # Rotate the x-axis labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_yticklabels(ax.get_yticks())
    plt.title('Confusion Matrix')
    plt.xlabel('Actual Class')
    plt.ylabel('Predicted Class')
    plt.show()

def plot_eigenvalues(top_eigenval):
    plt.figure(figsize=(8, 6))
    plt.bar(range(1, len(top_eigenval) + 1), top_eigenval, align='center')
    plt.xlabel('Eigenvalue Index')
    plt.ylabel('Eigenvalue')
    plt.title('Top Eigenvalues')
    plt.show()
    
def make_2pix_feat(ori_vec):
    combina = list(itertools.combinations(range(len(ori_vec)), 2))
    # result_arrays = ori_vec.tolist()
    result_arrays = []
    for comb in combina:
        dim1, dim2 = comb
        diff_array = ori_vec[dim1] - ori_vec[dim2]
        result_arrays.append(diff_array)
    return np.array(result_arrays)

# The mean (avg) image
mean_img = train_flat.mean(axis=0)
    
A_vec = train_flat - mean_img[np.newaxis, :]

# Way 1 (cannot use it PCA directly for recons!)
# cov_matrix = A_vec.dot(A_vec.T)/len(A_vec)

# Way 2
cov_matrix = A_vec.T.dot(A_vec)/len(A_vec)

eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
eigenvalues, eigenvectors = np.real(eigenvalues), np.real(eigenvectors)

# Sort eigenvalues and corresponding eigenvectors in descending order
sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_indices]
sorted_eigenvec = eigenvectors[:, sorted_indices]


# Count the number of positive elements
count_positive = np.sum(sorted_eigenvalues > 0)

# Compare eigenvalues, eigenvectors of 2 ways
# print(np.linalg.norm(diff_values))

# reconstruct single image
# pic1_re = mean_img + recons(A_vec[0], sorted_eigenvec, 100)

# ## BRUTE FORCE FOR BEST RESULT
combinations = itertools.product(range(150, 500, 10), range(5, 11), range(0, 3))

top_m = 130
# turn train images to projections (extract 2D to PC features),
train_projs = [np.dot(norm_img, sorted_eigenvec[:, :top_m])
               for norm_img in A_vec]
# train_2pixS = [make_2pix_feat(vec) for vec in train_projs]


# norm test data
test_norm = test_flat - mean_img[np.newaxis, :]
test_projs = [np.dot(norm_img, sorted_eigenvec[:, :top_m])
              for norm_img in test_norm]
# test_2pixS = [make_2pix_feat(vec) for vec in test_projs]


# reconstruct for all train set, calc recon error.

# train_recon = [mean_img + recons(norm_img, sorted_eigenvec, top_m)
#                for norm_img in A_vec]
# train_rec_err = np.mean(np.linalg.norm(train_flat-train_recon, ord=2, axis=1))


# learn KNN Classifier using train data
neigh = KNeighborsClassifier(3, weights='distance', metric='euclidean')
import time
start = time.time()
neigh.fit(train_projs, train_y)
print("train:", time.time() - start)

# Predict with KNN
start = time.time()
test_pred = neigh.predict(test_projs)
print("test:", time.time() - start)
test_acc = accuracy_score(test_y, test_pred)
print('KNN test acc: ', test_acc)  # M = 100, K = 3 gave best acc.
    
    
    