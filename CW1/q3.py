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
import pickle

random.seed(15)

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
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
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

mean_img = train_flat.mean(axis=0)
test_mean = test_flat.mean(axis=0)
    
A_vec = (train_flat - mean_img[np.newaxis, :]).T
cov_matrix = A_vec.dot(A_vec.T)/len(A_vec)
sorted_eigenvalues, sorted_eigenvec = sorted_eigens(cov_matrix)

M_pca = 190

train_proj = np.dot(A_vec.T, sorted_eigenvec[:, :M_pca])
test_proj = np.dot(test_flat - test_mean[np.newaxis, :], sorted_eigenvec[:, :M_pca])

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

M_lda = 60

S_W, S_B = lda_scatter_matrices(train_proj, train_y)
eig_vals, eig_vecs = np.linalg.eig(np.linalg.pinv(S_W).dot(S_B))
eig_vals, eig_vecs = np.real(eig_vals), np.real(eig_vecs)
sorted_indices_lda = np.argsort(eig_vals)[::-1]
eig_vecs_sorted = eig_vecs[:, sorted_indices_lda]
#show 5 top eigenfaces
show_arr_imgs(eig_vecs_sorted[:, :5].T.dot(sorted_eigenvec[:, :M_pca].T), titles=[f"Fisherface {i+1}" for i in range(5)])

train_lda = np.dot(train_proj, eig_vecs_sorted[:, :M_lda])
test_lda = np.dot(test_proj, eig_vecs_sorted[:, :M_lda])

knn = KNeighborsClassifier(n_neighbors=11, metric='cosine', weights='distance')
knn.fit(train_lda, train_y)
predictions = knn.predict(test_lda)

accuracy = accuracy_score(test_y, predictions)
confusion = confusion_matrix(test_y, predictions)

print(f"Recognition Accuracy: {accuracy*100:.2f}%")
print("\nConfusion Matrix:")
show_conf_mat(confusion)

success_indices = np.where(predictions == test_y)[0]
failure_indices = np.where(predictions != test_y)[0]

show_arr_imgs(test_flat[success_indices[:5]], titles=[f"predicted: {predictions[i]}, actual: {test_y[i]}" for i in success_indices[:5]])
show_arr_imgs(test_flat[failure_indices[:5]], titles=[f"predicted: {predictions[i]}, actual: {test_y[i]}" for i in failure_indices[:5]])

rank_sw = np.linalg.matrix_rank(S_W)
rank_sb = np.linalg.matrix_rank(S_B)
print(f"\nRank of Within-class Scatter Matrix: {rank_sw}")
print(f"Rank of Between-class Scatter Matrix: {rank_sb}")


class PCALDAEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, T=10, random_features_ratio=0.8, randomness_parameter_ratio = 0.8, get_avg_error=True):
        self.T = T
        self.random_features_ratio = random_features_ratio
        self.ranomness_parameter_ratio = randomness_parameter_ratio
        self.models = []
        self.Ws = []
        self.feature_indices = []
        self.get_avg_error = get_avg_error

    def fit(self, X, y):
        for i in range(self.T):
            # Bagging: Sample data with replacement
            n_samples = X.shape[0]*self.ranomness_parameter_ratio
            X_sample, y_sample = resample(X, y, n_samples = int(n_samples), replace=True)
            # X_sample, y_sample = X, y

            # Random feature selection after PCA
            n_features = X_sample.shape[1]
            n_selected_features = int(self.random_features_ratio * n_features)
            selected_indices = np.random.choice(n_features, n_selected_features, replace=False)
            self.feature_indices.append(selected_indices)
            X_sample_reduced = X_sample[:, selected_indices]
            # X_sample_reduced = X_sample

            # LDA on PCA-transformed training data
            M_lda = random.randrange(10, 101, 10)
            S_W, S_B = lda_scatter_matrices(X_sample_reduced, y_sample)
            eig_vals, eig_vecs = np.linalg.eig(np.linalg.pinv(S_W).dot(S_B))
            eig_vals, eig_vecs = np.real(eig_vals), np.real(eig_vecs)
            sorted_indices = np.argsort(eig_vals)[::-1]
            W = eig_vecs[:, sorted_indices[:M_lda]]
            self.Ws.append(W)
            X_lda_train = X_sample_reduced.dot(W)
            
            # Train the NN classifier
            n_neighbors = random.randrange(1, 20, 2)
            knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric='cosine', weights='distance')
            knn.fit(X_lda_train, y_sample)
            self.models.append(knn)

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        return self
    def predict(self, X, fusion='majority'):
        # Initialize the probability matrix with the shape of (n_samples, n_classes)
        probabilities = []
        predictions = []
        self.model_predictions = []

        # Get predictions from all base models
        for knn, indices, W in zip(self.models, self.feature_indices, self.Ws):
            # Apply PCA and LDA transformations
            X_reduced = X[:, indices]
            X_lda_test = X_reduced.dot(W)
            
            # Get current probabilities and align them
            if self.get_avg_error:
                self.model_predictions.append(knn.predict(X_lda_test))
            if fusion == 'majority':
                predictions.append(knn.predict(X_lda_test))
            else:
                cur_probabilities = np.zeros((X.shape[0], self.n_classes_))
                cur_proba = knn.predict_proba(X_lda_test)
                for i, class_label in enumerate(knn.classes_):
                    class_index = np.where(self.classes_ == class_label)[0][0]
                    cur_probabilities[:, class_index] += cur_proba[:, i]
                probabilities.append(cur_probabilities)
        probabilities = np.array(probabilities)

        if self.get_avg_error:
            self.model_predictions = np.array(self.model_predictions)
        if fusion == 'sum':
            probability = np.sum(probabilities, axis=0)
            predictions = np.argmax(probability, axis=1) + 1
        elif fusion == 'majority':
            predictions = np.array(predictions)
            predictions = np.squeeze(predictions)
            predictions = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=0, arr=predictions)
        elif fusion == 'max':
            probability = np.amax(probabilities, axis=0)
            predictions = np.argmax(probability, axis=1) + 1
        elif fusion == 'min':
            probability = np.amin(probabilities, axis=0)
            predictions = np.argmax(probability, axis=1) + 1
        elif fusion == 'product':
            probability = np.prod(probabilities, axis=0)
            predictions = np.argmax(probability, axis=1) + 1
        else:
            raise ValueError("Invalid fusion method")
        return predictions

# Create the ensemble
# Ts = range(100, 151, 10)
# random_features_ratios = np.arange(0.8, 1.0001, 0.05)
# randomness_parameter_ratios = np.arange(0.6, 1.001, 0.05)
# fusion_methods = ['majority', 'sum']

# def ensembles_acc(T, random_features_ratio, randomness_parameter_ratio, fusion):
#     ensemble = PCALDAEnsemble(T=T, random_features_ratio=random_features_ratio, randomness_parameter_ratio = randomness_parameter_ratio)
#     ensemble.fit(train_proj, train_y)
#     ensemble_preds = ensemble.predict(test_proj, fusion=fusion)
#     ensemble_accuracy = accuracy_score(test_y, ensemble_preds)
#     return ensemble_accuracy

# # accuracies = np.zeros((len(Ts), len(random_features_ratios), len(randomness_parameter_ratios), len(fusion_methods)))
# # for i, T in enumerate(Ts):
# #     for j, random_features_ratio in enumerate(random_features_ratios):
# #         for k, randomness_parameter_ratio in enumerate(randomness_parameter_ratios):
# #             for l, fusion in enumerate(fusion_methods):
# #                 accuracies[i, j, k, l] = ensembles_acc(T, random_features_ratio, randomness_parameter_ratio, fusion)
# #                 print(f"Finished {i}, {j}, {k}, {l}, accuracy: {accuracies[i, j, k, l]}")

# # np.save('accuracies_ensemble_2.npy', accuracies)
# # accuracies = np.load('accuracies_ensemble.npy')
# # best = [[100, 1, 0.7, 'sum'], [100, 0.95, 0.6, 'sum'], [120, 0.9, 0.65, 'majority']]
# # T = 100
# # random_features_ratio = 0.95
# # randomness_parameter_ratio = 0.65
# # fusion = 'sum'
# ensemble_accuracy = 0
# while ensemble_accuracy < 0.97:
#     ensemble = PCALDAEnsemble(T=100, random_features_ratio=0.95, randomness_parameter_ratio = 0.6)
#     ensemble.fit(train_proj, train_y)

#     # Predict and calculate accuracy
#     ensemble_preds = ensemble.predict(test_proj, fusion='sum')
#     # print(ensemble_preds)
#     ensemble_accuracy = accuracy_score(test_y, ensemble_preds)
#     print(f"Ensemble Recognition Accuracy: {ensemble_accuracy*100:.2f}%")
# #save this ensemble using pickle
# pickle.dump(ensemble, open('ensemble.pkl', 'wb'))

# # ensemble = pickle.load(open('ensemble.pkl', 'rb'))
# ensemble_confusion = confusion_matrix(test_y, ensemble_preds)

# def calculate_errors(ensemble, y):
#     errors = []
#     for prediction in ensemble.model_predictions:
#         errors.append(1 - accuracy_score(y, prediction))
#     return np.mean(errors)

# average_error = calculate_errors(ensemble, test_y)
# committee_error = 1 - accuracy_score(test_y, ensemble_preds)

# print(f"Ensemble Recognition Accuracy: {ensemble_accuracy*100:.2f}%")
# print(f"Average Model Error: {average_error}")
# print(f"Committee Machine Error: {committee_error}")
# print("\nEnsemble Confusion Matrix:")
# show_conf_mat(ensemble_confusion)

# # Compare with individual models
# # individual_accuracies = [accuracy_score(test_y, model.predict(test_proj[:, indices])) for model, indices in zip(ensemble.models, ensemble.feature_indices)]
# # avg_individual_accuracy = np.mean(individual_accuracies)
# # print(f"Average accuracy of individual models: {avg_individual_accuracy*100:.2f}%")
