%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% svm_train.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; clc; close all;
rand('seed', 123);
[sparseTrainMatrix, tokenlist, trainCategory] = ...
readMatrix('MATRIX.TRAIN');
%readMatrix(sprintf('MATRIX.TRAIN.%d', num_train)) 
% use this for different training file

Xtrain = full(sparseTrainMatrix);
m_train = size(Xtrain, 1);
ytrain = (2 * trainCategory - 1)'; % convert labels to {-1, +1}
Xtrain = 1.0 * (Xtrain > 0); % ignore frequency, only care about occurence or not

squared_X_train = sum(Xtrain.^2, 2); % number of words in each example
gram_train = Xtrain * Xtrain'; % Gram matrix
tau = 8;

% Get full training matrix for kernels using vectorized code.
% Basically, exp(-(norm(xi)^2 - 2*dot(xi,xj) + norm(xj)^2)) / (2*sigma^2)
% Don't see a better way to implement this.
Ktrain = full(exp(-(repmat(squared_X_train, 1, m_train) ...
                    + repmat(squared_X_train', m_train, 1) ...
                    - 2 * gram_train) / (2 * tau^2)));

lambda = 1 / (64 * m_train);
num_outer_loops = 40;
alpha = zeros(m_train, 1);

avg_alpha = zeros(m_train, 1);


count = 0;
for ii = 1:(num_outer_loops * m_train)
  count = count + 1;
  ind = ceil(rand * m_train); % Pick a random sample
  
  % compute margin
  % y[i]*(w'*x[i]+b) >= 1 {can ignore b}
  % w'*x[i]= sum((alpha*y*x)'*x[i])
  % since using RBF kernel, w'*x[i]= sum(y*K(x[i],x)*alpha)
  % since no y in w'*x[i], alpha can be < 0 instead of 0 < alpha < C
    margin = ytrain(ind) * Ktrain(ind, :) * alpha;
  
  % compute gradient. del(psi)/del(alpha)=0. 
  g = -(margin < 1) * ytrain(ind) * Ktrain(:, ind) + ...
      m_train * lambda * (Ktrain(:, ind) * alpha(ind));
  
%   g(ind) = g(ind) + m_train * lambda * Ktrain(ind,:) * alpha;

  alpha = alpha - g / sqrt(count); % scale gradient and add to alpha
  
  avg_alpha = avg_alpha + alpha; %actually cumulative alpha in the loop
  
end
avg_alpha = avg_alpha / (num_outer_loops * m_train); % actual average
