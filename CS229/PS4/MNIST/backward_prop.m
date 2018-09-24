function [dW1, db1, dW2, db2] = backward_prop(X, y_onehot, W1, b1, W2, b2, lambda)
%% backward propagation for our 1 layer network
%% input parameters
% X is our m x n dataset, where m = number of samples, n = number of
% features
% y is the length m label vector for each sample
% W1 is our h1 x n weight matrix, where h1 = number of hidden units in
% layer 1
% b1 is the length h1 column vector of bias terms associated with layer 1
% W2 is the c x h1 weight matrix, where c = number of classes
% b2 is the length h2 column vector of bias terms associated with the output
%% output parameters
% returns the gradient of W1, b1, W2, b2 as dW1, db1, dW2, db2
%% Your code here
A1 = sigmoid_func(X*W1'+b1');
A2 = softmax_func(A1*W2'+b2');
dW2 = (A2 - y_onehot)'*A1./size(X,1) + 2*lambda*W2;
db2 = mean(A2 - y_onehot, 1)';
dW1 = ((A2 - y_onehot)*W2.*(A1.*(1-A1)))'*X./size(X,1)...
    + 2*lambda*W1;
db1 = mean((A2 - y_onehot)*W2.*(A1.*(1-A1)), 1)';
end