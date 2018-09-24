function [ H ] = logistic_Hess( X, t )
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
m = size(X,1);
% H = 1/m*(X'*X)*( sigmoid(X*t)'*(1 - sigmoid(X*t)) );
H = 1/m*(X'*X)*( sigmoid(X*t)'*(1 - sigmoid(X*t)) );
end

