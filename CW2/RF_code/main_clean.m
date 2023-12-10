% This script aims to help the student in understanding the main ideas behind
% random forests and how they can be implemented, being a tool to help the
% on their first coursework on "Selected Topics in Computer Vision" 2018-2019.
% 
% The script is:
%   
% Experiment with Caltech dataset for image categorisation (Intro to Coursework 1)
% 
% Instructions: 
%   - Run the different sections in order (some sections require variables
%   from previous sections)
%   - Try to understand the code and how it relates to theory.
%   - Play with different forest parameters and understand their impact.
% 
% The script is based in:
% Simple Random Forest Toolbox for Matlab
%   written by Mang Shao and Tae-Kyun Kim, June 20, 2014.
%   updated by Tae-Kyun Kim, Feb 09, 2017
%   updated by G. Garcia-Hernando, Jan 10, 2018

% Last update: January 2019

% The codes are made for educational purposes only.
% Some parts are inspired by Karpathy's RF Toolbox
% Under BSD Licence

clear all; close all; 
% Initialisation
init; clc;

param.num = 120;     % number of trees
param.depth = 13;    % trees depth
param.splitNum = 6; % Number of trials in split function
param.split = 'IG'; % Currently support 'information gain' only

% Complete getData.m by writing your own lines of code to obtain the visual 
% vocabulary and the bag-of-words histograms for both training and testing data. 
% You can use any existing code for K-means (note different codes require different memory and computation time).

[data_train, data_test] = getData('Caltech');
% [data_train, data_test] = getData_RFcodebook('Caltech');
% data_test = data_train;

close all;

% Train your random forest with the BoW histograms.
tic
[trees, ~] = growTrees(data_train,param);
toc

% Test the random forest
tic
testTrees_script;
toc

% Show the results.
confus_script;
