function [ h ] = sigmoid( z )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
h = 1./( ones(size(z)) + exp(-z) );

end

