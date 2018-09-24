clear; clc; close all;

% DATASET FOR PS2(Q1)
% D=load('data_b.txt');
% y=D(:,1);
% X=D(:,2:end);
% X=[ones(length(y),1) X];

y = load('logistic_y.txt');
X = load('logistic_x.txt');
X=[ones(length(y),1) X];
t = zeros(3,1);

% CONVERT -1 TO 0
y(find(y==-1))= 0;

% ROOT FINDING METHOD, FINDS WHERE del_J/del_Theta EQUALS 0
% THAT IS WHERE COST FUNCTION J(Theta) IS MINIMUM
% Newton-Rhapson
% i = 0;
% while( norm(logistic_grad(X, y, t)) > 10e-12)
%     i = i + 1;
%     t = t - inv(logistic_Hess(X, t))*logistic_grad(X, y, t);  
% end

% MINIMISE (fx - y) WITH NEWTON'S, EE103 METHOD 
% i = 0;
% while( norm(sigmoid(X*t)-y) > 10e-12) % TOO LARGE TO CONVERGE, NEED A SMALL STOPPING CRITERIA
%     i = i + 1;
%     t = t - pinv( sigmoid(X*t)'*( ones(length(y),1)-sigmoid(X*t) )*X )*...
%         (sigmoid(X*t)-y);
% end

% Gradient Descent
i = 0; a = 1;
% USES del_J/del_Theta FOR CONVERGENCE
while( norm(logistic_grad(X, y, t)) > 10e-12 )
    i = i + 1;
    t = t - a*logistic_grad(X, y, t);  
end

% USES diff_J FOR CONVERGENCE
% while true
%     i = i + 1;
%     tp=t;
%     t = t - a*logistic_grad(X, y, t);
%     J=y'*log(sigmoid(X*t))+(ones(length(y),1)-y)'*log(ones(length(y),1)-sigmoid(X*t));
%     Jp=y'*log(sigmoid(X*tp))+(ones(length(y),1)-y)'*log(ones(length(y),1)-sigmoid(X*tp));
%     if( abs(J-Jp) <= 10e-12 ) % CHECK DIFFERENCE OF J FOR CONVERGENCE
%         break;
%     end
% end

Xpos = X(find(y == 1), 2:3);
Xneg = X(find(y == 0), 2:3);

figure;
plot(Xpos(:,1),Xpos(:,2),'bo',Xneg(:,1),Xneg(:,2),'r+');
hold on;

x1 = 0:0.1:8;
x2 = -(x1.*t(2)+t(1))./t(3);
plot(x1, x2, 'k');
