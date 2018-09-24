function [ J ] = logistic_cost( X, y, t )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
m = size(y,1);
J = -1/m*( y'*log(sigmoid(X*t)) + ...
    ( ones(size(y)) - y )'*log( ones(size(y)) - sigmoid(X*t) ) );

end

