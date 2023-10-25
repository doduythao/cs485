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
import time

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

def sorted_eigens(cov_matrix):
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvec = eigenvectors[:, sorted_indices]
    return sorted_eigenvalues, sorted_eigenvec

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

start_time = time.time()
mean_img = train_flat.mean(axis=0)
    
A_vec = train_flat - mean_img[np.newaxis, :]

# Way 1 (cannot use it PCA directly for recons!)
# cov_matrix = A_vec.dot(A_vec.T)/len(A_vec)

# Way 2
cov_matrix = A_vec.T.dot(A_vec)/len(A_vec)

sorted_eigenvalues, sorted_eigenvec = sorted_eigens(cov_matrix)
# eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
# # eigenvalues, eigenvectors = np.real(eigenvalues), np.real(eigenvectors)

# # Sort eigenvalues and corresponding eigenvectors in descending order
# sorted_indices = np.argsort(eigenvalues)[::-1]
# sorted_eigenvalues = eigenvalues[sorted_indices]
# sorted_eigenvec = eigenvectors[:, sorted_indices]


# Count the number of positive elements
count_positive = np.sum(sorted_eigenvalues > 0)

# Compare eigenvalues, eigenvectors of 2 ways
# print(np.linalg.norm(diff_values))

# reconstruct single image
# pic1_re = mean_img + recons(A_vec[0], sorted_eigenvec, 100)

# ## BRUTE FORCE FOR BEST RESULT
combinations = itertools.product(range(150, 500, 10), range(5, 11), range(0, 3))

top_m = 100
# turn train images to projections (extract 2D to PC features),
train_projs = [np.dot(norm_img, sorted_eigenvec[:, :top_m])
               for norm_img in A_vec]
# train_2pixS = [make_2pix_feat(vec) for vec in train_projs]


# norm test data
test_norm = test_flat - mean_img[np.newaxis, :]
test_projs = [np.dot(norm_img, sorted_eigenvec[:, :top_m])
              for norm_img in test_norm]
end_time = time.time()
print(f'Time elapsed (Batch PCA): {end_time-start_time}s')
# test_2pixS = [make_2pix_feat(vec) for vec in test_projs]


# reconstruct for all train set, calc recon error.

train_recon = [mean_img + recons(norm_img, sorted_eigenvec, top_m)
               for norm_img in A_vec]
train_rec_err = np.mean(np.linalg.norm(train_flat-train_recon, ord=2, axis=1))
# show_arr_imgs(train_recon[:5])

# # learn KNN Classifier using train data
# n_nb = 3
# neigh = KNeighborsClassifier(n_nb, weights='distance', metric='cosine')
# neigh.fit(train_projs, train_y)

# # Predict with KNN
# test_pred = neigh.predict(test_projs)
# test_acc = accuracy_score(test_y, test_pred)
# print('KNN test acc: ', test_acc)  # M = 100, K = 3 gave best acc.
# print(precision_recall_fscore_support(test_y, test_pred, average='macro'))

# # Calculate the confusion matrix
# confusion_matrix = confusion_matrix(test_y, test_pred)
# show_conf_mat(confusion_matrix)


### Q2 - Incremental PCA

# Split training data into n_sub subsets
n_sub = 4
train_subsets = np.array(np.array_split(train_flat, n_sub))

models = []
# print('Training incremental PCA models...')
start_time = time.time()
for i in range(n_sub):
    subset_mean = train_subsets[i].mean(axis=0)
    subset_N = len(train_subsets[i])

    subset_A_vec = train_subsets[i] - subset_mean[np.newaxis, :]
    subset_cov_matrix = subset_A_vec.T.dot(subset_A_vec)/subset_N
    # subset_eigenvalues, subset_eigenvectors = np.linalg.eigh(subset_cov_matrix)

    # subset_sorted_indices = np.argsort(subset_eigenvalues)[::-1]
    # subset_sorted_eigenvalues = subset_eigenvalues[subset_sorted_indices]
    # subset_sorted_eigenvec = subset_eigenvectors[:, subset_sorted_indices]
    if i==0:
        s1_pause_time = time.time()
    models.append((subset_mean, subset_N, subset_cov_matrix))

incremental_models = [models[0]]

for i in range(n_sub-1):
    inc_N = incremental_models[-1][1] + models[i+1][1]
    inc_mean = (incremental_models[-1][0]*incremental_models[-1][1] + models[i+1][0]*models[i+1][1])/inc_N
    inc_cov_matrix = incremental_models[-1][1]/inc_N * incremental_models[-1][2] + models[i+1][1]/inc_N * models[i+1][2] + \
            (incremental_models[-1][1]*models[i+1][1]/(inc_N**2)) * (incremental_models[-1][0]-models[i+1][0]) * (incremental_models[-1][0]-models[i+1][0]).T
    incremental_models.append((inc_mean, inc_N, inc_cov_matrix))

# models[0] is PCA trained only by the first subset:
inc_pause_time = time.time()
s1_mean, s1_N, s1_cov_matrix = models[0]
s1_resume_time = time.time()
s1_sorted_eigenvalues, s1_sorted_eigenvec = sorted_eigens(s1_cov_matrix)
s1_A_vec = train_subsets[0] - s1_mean[np.newaxis, :]
s1_train_projs = [np.dot(norm_img, s1_sorted_eigenvec[:, :top_m])
               for norm_img in s1_A_vec]

# print(s1_train_projs[0].shape, len(s1_train_projs))
s1_test_norm = test_flat - s1_mean[np.newaxis, :]
s1_test_projs = [np.dot(norm_img, s1_sorted_eigenvec[:, :top_m])
              for norm_img in s1_test_norm]
s1_end_time = time.time()
print(f'Time elapsed (1st subset PCA): {s1_end_time-s1_resume_time+s1_pause_time-start_time}s')

# reconstruct for all train set, calc recon error.
s1_train_recon = [s1_mean + recons(norm_img, s1_sorted_eigenvec, top_m)
               for norm_img in A_vec]
s1_train_rec_err = np.mean(np.linalg.norm(train_flat-s1_train_recon, ord=2, axis=1))
# show_arr_imgs(train_flat[:5])
# show_arr_imgs(s1_train_recon[:5])


# incremental_models[-1] is PCA trained by all subsets:
inc_resume_time = time.time()
inc_mean, inc_N, inc_cov_matrix = incremental_models[-1]
inc_sorted_eigenvalues, inc_sorted_eigenvec = sorted_eigens(inc_cov_matrix)
inc_A_vec = train_flat - inc_mean[np.newaxis, :]
inc_train_projs = [np.dot(norm_img, inc_sorted_eigenvec[:, :top_m])
               for norm_img in inc_A_vec]

inc_test_norm = test_flat - inc_mean[np.newaxis, :]
inc_test_projs = [np.dot(norm_img, inc_sorted_eigenvec[:, :top_m])
              for norm_img in inc_test_norm]
end_time = time.time()
print(f'Time elapsed (Incremental PCA): {end_time-inc_resume_time+inc_pause_time-start_time}s')

inc_train_recon = [inc_mean + recons(norm_img, inc_sorted_eigenvec, top_m)
                for norm_img in A_vec]
inc_train_rec_err = np.mean(np.linalg.norm(train_flat-inc_train_recon, ord=2, axis=1))
# show_arr_imgs(inc_train_recon[:5])

print("Reconstruction errors:")
print(f'Batch PCA: {train_rec_err}')
print(f'1st subset PCA: {s1_train_rec_err}')
print(f'Incremental PCA: {inc_train_rec_err}')



# Random Forest. Then learn to fit
rfc = RandomForestClassifier(n_estimators=360, 
                              max_depth=10, 
                              max_features = 'sqrt', 
                              random_state=0)
rfc.fit(train_projs, train_y)

# Predict with RF
test_pred = rfc.predict(test_projs)
test_acc = accuracy_score(test_y, test_pred)
    
    
    