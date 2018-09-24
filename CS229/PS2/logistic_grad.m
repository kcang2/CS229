function [ d ] = logistic_grad( X, y, t )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
m = size(y,1);
d = 1/m.*X'*( sigmoid(X*t) - y );

end

