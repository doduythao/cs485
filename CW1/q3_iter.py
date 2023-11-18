# -*- coding: utf-8 -*-
"""
CW 1 of CS485
"""
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import resample
import numpy as np
import seaborn as sns
import itertools
from matplotlib import pyplot as plt
import time
import random

random.seed(0)

import scipy.io
mat = scipy.io.loadmat('face.mat')
data = mat['X'].reshape(2576, 52, 10)
Y = mat['l'].reshape(52, 10)



train, test = data[:, :, :8], data[:, :, 8:]  # 8:2 Split.
# 1st dim is each image, 2nd dim is each person.

train_flat, test_flat = train.reshape(2576, 52*8).T, test.reshape(2576, 52*2).T
train_y, test_y = Y[:, :8].flatten(), Y[:, 8:].flatten()

# Visualize the images
def show_arr_imgs(arr, titles=None):
    fig, axes = plt.subplots(1, len(arr), figsize=(20, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(arr[i].reshape(46, 56).T, cmap='gray')
        if titles is not None:
            ax.set_title(titles[i])
        else:
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

def recons_ori(inp_norm, eigen_vec, M_pca):
    out = 0
    for i in range(0, M_pca):
        weight = np.dot(inp_norm, eigen_vec[:,i])
        out += eigen_vec[:,i] * weight
    return out

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

def recons(inp_norm, eigen_vec, M_pca):
    weights = np.dot(inp_norm, eigen_vec[:, :M_pca])
    out = np.dot(weights, eigen_vec[:, :M_pca].T)
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


def lda_scatter_matrices(X, y):
    # Calculate the overall mean
    mean_overall = np.mean(X, axis=0)
    
    # Calculate the mean of each class
    classes = np.unique(y)
    mean_vectors = []
    for cl in classes:
        mean_vectors.append(np.mean(X[y == cl], axis=0))
    
    # Within-class scatter matrix
    S_W = np.zeros((X.shape[1], X.shape[1]))
    for cl, mv in zip(classes, mean_vectors):
        class_sc_mat = np.zeros((X.shape[1], X.shape[1]))
        for row in X[y == cl]:
            row, mv = row.reshape(X.shape[1], 1), mv.reshape(X.shape[1], 1)
            class_sc_mat += (row - mv).dot((row - mv).T)
        S_W += class_sc_mat
    
    # Between-class scatter matrix
    S_B = np.zeros((X.shape[1], X.shape[1]))
    for mv in mean_vectors:
        N = X[y == cl].shape[0]
        mv = mv.reshape(X.shape[1], 1)
        mean_overall = mean_overall.reshape(X.shape[1], 1)
        S_B += N * (mv - mean_overall).dot((mv - mean_overall).T)
        
    return S_W, S_B

mean_img = train_flat.mean(axis=0)
test_mean = test_flat.mean(axis=0)
    
A_vec = (train_flat - mean_img[np.newaxis, :]).T
cov_matrix = A_vec.dot(A_vec.T)/len(A_vec)
sorted_eigenvalues, sorted_eigenvec = sorted_eigens(cov_matrix)

def PCALDA(M_pca=100, M_lda=50, k=1, method='euclidean'):
    train_proj = np.dot(A_vec.T, sorted_eigenvec[:, :M_pca])
    test_proj = np.dot(test_flat - test_mean[np.newaxis, :], sorted_eigenvec[:, :M_pca])

    S_W, S_B = lda_scatter_matrices(train_proj, train_y)
    eig_vals, eig_vecs = np.linalg.eig(np.linalg.pinv(S_W).dot(S_B))
    eig_vals, eig_vecs = np.real(eig_vals), np.real(eig_vecs)
    sorted_indices_lda = np.argsort(eig_vals)[::-1]
    eig_vecs_sorted = eig_vecs[:, sorted_indices_lda]

    train_lda = np.dot(train_proj, eig_vecs_sorted[:, :M_lda])
    test_lda = np.dot(test_proj, eig_vecs_sorted[:, :M_lda])

    knn = KNeighborsClassifier(n_neighbors=k, metric=method, weights='distance')
    knn.fit(train_lda, train_y)
    predictions = knn.predict(test_lda)

    accuracy = accuracy_score(test_y, predictions)
    # confusion = confusion_matrix(test_y, predictions)
    return accuracy

#TODO - Iteration of M_lda and M_pca, k and NN method for best recognition accuracy

M_pcas = range(100, 251, 10)
M_ldas = range(10, 101, 10)
ks = range(1, 21, 2)
progress = 0
methods = ['euclidean', 'manhattan', 'cosine']

accuracies = np.zeros((len(M_pcas), len(M_ldas), len(ks), len(methods)))

# for method in methods:
#     for M_pca in M_pcas:
#         for M_lda in M_ldas:
#             for k in ks:
#                 progress += 1
#                 accuracy = PCALDA(M_pca=M_pca, M_lda=M_lda, k=k, method=method)
#                 accuracies[M_pca//10 - 10, M_lda//10 - 1, k//2, methods.index(method)] = accuracy
#                 print(f'M_pca: {M_pca}, M_lda: {M_lda}, k: {k}, method: {method}, accuracy: {accuracy}, progress: {progress}/600')

#save accuracies to file
# np.save('accuracies.npy', accuracies)
accuracies = np.load('accuracies.npy', allow_pickle=True)
# to csv, only cosine and different k in different files
for k in ks:
    np.savetxt(f'k{k}.csv', accuracies[:, :, k//2, 2].T, delimiter=',')

average_method_accuracies = np.mean(accuracies[:, :, :, 2], axis=(0, 1))
print("average_method_accuracies: ", average_method_accuracies)

max_accuracy_index = np.argmax(accuracies)

# Convert this index into a tuple of indices corresponding to the 4D shape of the accuracies array
max_accuracy_indices = np.unravel_index(max_accuracy_index, accuracies.shape)
M_pca_max = M_pcas[max_accuracy_indices[0]]
M_lda_max = M_ldas[max_accuracy_indices[1]]
k_max = ks[max_accuracy_indices[2]]
method_max = methods[max_accuracy_indices[3]]

# Print the parameters with the highest accuracy
print(f'Maximum accuracy parameters: M_pca: {M_pca_max}, M_lda: {M_lda_max}, k: {k_max}, method: {method_max}')
print(f'Maximum accuracy: {accuracies[max_accuracy_indices]:.2f}')






# class PCALDAEnsemble(BaseEstimator, ClassifierMixin):
#     def __init__(self, T=10, random_features_ratio=0.8, M_lda=50, randomness_parameter_ratio = 0.8):
#         self.T = T
#         self.random_features_ratio = random_features_ratio
#         self.M_lda = M_lda
#         self.ranomness_parameter_ratio = randomness_parameter_ratio
#         self.models = []
#         self.Ws = []
#         self.feature_indices = []

#     def fit(self, X, y):
#         for i in range(self.T):
#             # Bagging: Sample data with replacement
#             n_samples = X.shape[0]*self.ranomness_parameter_ratio
#             X_sample, y_sample = resample(X, y, n_samples = int(n_samples), replace=True)
#             # X_sample, y_sample = X, y

#             # Random feature selection after PCA
#             n_features = X_sample.shape[1]
#             n_selected_features = int(self.random_features_ratio * n_features)
#             selected_indices = np.random.choice(n_features, n_selected_features, replace=False)
#             self.feature_indices.append(selected_indices)
#             X_sample_reduced = X_sample[:, selected_indices]
#             # X_sample_reduced = X_sample

#             # LDA on PCA-transformed training data
#             S_W, S_B = lda_scatter_matrices(X_sample_reduced, y_sample)
#             eig_vals, eig_vecs = np.linalg.eig(np.linalg.pinv(S_W).dot(S_B))
#             eig_vals, eig_vecs = np.real(eig_vals), np.real(eig_vecs)
#             sorted_indices = np.argsort(eig_vals)[::-1]
#             W = eig_vecs[:, sorted_indices[:self.M_lda]]
#             self.Ws.append(W)
#             X_lda_train = X_sample_reduced.dot(W)
            
#             # Train the NN classifier
#             knn = KNeighborsClassifier(n_neighbors=21)
#             knn.fit(X_lda_train, y_sample)
#             self.models.append(knn)

#         self.classes_ = np.unique(y)
#         self.n_classes_ = len(self.classes_)
#         return self
#     def predict(self, X, fusion='majority'):
#         # Initialize the probability matrix with the shape of (n_samples, n_classes)
#         probabilities = []
#         predictions = []

#         # Get predictions from all base models
#         for knn, indices, W in zip(self.models, self.feature_indices, self.Ws):
#             # Apply PCA and LDA transformations
#             X_reduced = X[:, indices]
#             X_lda_test = X_reduced.dot(W)
            
#             # Get current probabilities and align them
#             if fusion == 'majority':
#                 predictions.append(knn.predict(X_lda_test))
#             else:
#                 cur_probabilities = np.zeros((X.shape[0], self.n_classes_))
#                 cur_proba = knn.predict_proba(X_lda_test)
#                 for i, class_label in enumerate(knn.classes_):
#                     class_index = np.where(self.classes_ == class_label)[0][0]
#                     cur_probabilities[:, class_index] += cur_proba[:, i]
#                 probabilities.append(cur_probabilities)
#         probabilities = np.array(probabilities)

#         if fusion == 'sum':
#             probability = np.sum(probabilities, axis=0)
#             predictions = np.argmax(probability, axis=1) + 1
#         elif fusion == 'majority':
#             predictions = np.array(predictions)
#             predictions = np.squeeze(predictions)
#             predictions = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=0, arr=predictions)
#         elif fusion == 'max':
#             probability = np.amax(probabilities, axis=0)
#             predictions = np.argmax(probability, axis=1) + 1
#         elif fusion == 'min':
#             probability = np.amin(probabilities, axis=0)
#             predictions = np.argmax(probability, axis=1) + 1
#         elif fusion == 'product':
#             probability = np.prod(probabilities, axis=0)
#             predictions = np.argmax(probability, axis=1) + 1
#         else:
#             raise ValueError("Invalid fusion method")
#         return predictions

# # Create the ensemble
# ensemble = PCALDAEnsemble(T=100, random_features_ratio=0.9, M_lda=50)
# ensemble.fit(train_proj, train_y)

# # Predict and calculate accuracy
# fusion_methods = ['majority', 'sum', 'product', 'max', 'min']
# ensemble_preds = ensemble.predict(test_proj, fusion='majority')
# print(ensemble_preds)

# ensemble_accuracy = accuracy_score(test_y, ensemble_preds)
# ensemble_confusion = confusion_matrix(test_y, ensemble_preds)

# print(f"Ensemble Recognition Accuracy: {ensemble_accuracy*100:.2f}%")
# print("\nEnsemble Confusion Matrix:")
# show_conf_mat(ensemble_confusion)

# # Compare with individual models
# # individual_accuracies = [accuracy_score(test_y, model.predict(test_proj[:, indices])) for model, indices in zip(ensemble.models, ensemble.feature_indices)]
# # avg_individual_accuracy = np.mean(individual_accuracies)
# # print(f"Average accuracy of individual models: {avg_individual_accuracy*100:.2f}%")
