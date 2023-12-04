rng("default");

clear all; close all; 
% Initialisation
init; clc;
param.num = 200;     % number of trees
param.depth = 15;    % trees depth
param.splitNum = 10; % Number of trials in split function
param.split = 'IG'; % Currently support 'information gain' only

% Complete getData.m by writing your own lines of code to obtain the visual 
% vocabulary and the bag-of-words histograms for both training and testing data. 
% You can use any existing code for K-means (note different codes require different memory and computation time).

[data_train, data_test] = getData('Caltech');
close all;

% Train your random forest with the BoW histograms.
tic
trees = growTrees(data_train,param);
toc

% Test the random forest
tic
testTrees_script;
toc

% Show the results.
confus_script;