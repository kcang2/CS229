  function loss = cross_entropy(pred, y_onehot)
%% function for computing cross entropy loss
%% input parameters: 
% pred and y_onehot are both m x c matrix, the element at position (i j)
% corresponds to the probability and actual label of sample i for class j.
%% output: average cross entropy loss
%% Your code here
loss = -mean(sum(y_onehot.*log(pred),2));
end