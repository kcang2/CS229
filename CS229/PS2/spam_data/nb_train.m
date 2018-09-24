clear;clc;close all;
[spmatrix, tokenlist, trainCategory] = readMatrix('MATRIX.TRAIN');

trainMatrix = full(spmatrix);
numTrainDocs = size(trainMatrix, 1);
numTokens = size(trainMatrix, 2);

% trainMatrix is now a (numTrainDocs x numTokens) matrix.
% Each row represents a unique document (email).
% The j-th column of the row $i$ represents the number of times the j-th
% token appeared in email $i$. 

% tokenlist is a long string containing the list of all tokens (words).
% These tokens are easily known by position in the file TOKENS_LIST

% trainCategory is a (1 x numTrainDocs) vector containing the true 
% classifications for the documents just read in. The i-th entry gives the 
% correct class for the i-th email (which corresponds to the i-th row in 
% the document word matrix).

% Spam documents are indicated as class 1, and non-spam as class 0.
% Note that for the SVM, you would want to convert these to +1 and -1.


% YOUR CODE HERE
% unik=unique(trainMatrix);
phi=zeros(2,numTokens+1);

spam_index=find(trainCategory==1);
notspam_index=find(trainCategory==0);
num_spam=length(spam_index);
num_notspam=length(notspam_index);

phi(1,end)=log(num_spam/numTrainDocs);
phi(2,end)=log(num_notspam/numTrainDocs);

spam_matrix=trainMatrix(spam_index,:);
notspam_matrix=trainMatrix(notspam_index,:);
spam_words=sum(sum(spam_matrix));
notspam_words=sum(sum(notspam_matrix));

phi(1,1:end-1)=log((sum(spam_matrix,1)+1)./(spam_words+numTokens));
phi(2,1:end-1)=log((sum(notspam_matrix,1)+1)./(notspam_words+numTokens));

phi=phi';

% most indicative spam words
% logratio=(phi(1,1:end-1)-phi(2,end-1));
% [~,indic]=sort(logratio,'descend');
% tokenarray=strsplit(tokenlist);
% mostspam=tokenarray(indic(1:5));

