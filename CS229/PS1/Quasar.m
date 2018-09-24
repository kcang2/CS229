clear; clc; close all;

train = load('quasar_train.csv');
lambda = train(1,:)';
test = load('quasar_test.csv');
x = [ ones(length(lambda),1) lambda];
ytrain = train(2:end,:);
ytest = test(2:end,:);

ytrain1 = ytrain(1,:)';
% theta1 = (x'*x)\(x'*ytrain1);
% f1train = x*theta1;
% figure;
% plot(lambda,f1train); hold on; plot(lambda,ytrain1);

% t=[1,5,10,100,1000];
%        
% for i=1:length(t)
%     f1wtrain=zeros(length(lambda),1);
%     for j=1:length(lambda)
%         w = diag(exp(-(lambda(j)-lambda).^2/(2*(t(i)^2))));
%         theta1w = (x'*w*x)\(x'*w*ytrain1);
%         f1wtrain(j,1) = x(j,:)*theta1w;
%     end
%     figure;
%     plot(lambda,f1wtrain); hold on; plot(lambda,ytrain1);
% end

tsmooth=5;
w = exp(-(lambda'-lambda).^2/(2*(tsmooth^2)));

if (~exist('smoothed_y_train.mat'))
    ywtrain=zeros(size(ytrain));
    for i=1:size(ytrain,1)
       for  j=1:length(lambda)
            theta = (x'*diag(w(:,j))*x)\(x'*diag(w(:,j))*ytrain(i,:)');
            ywtrain(i,j)=x(j,:)*theta;
       end  
    end
    save('smoothed_y_train.mat','ywtrain');
else
    load('smoothed_y_train.mat','ywtrain');
end

if (~exist('smoothed_y_test.mat'))
    ywtest=zeros(size(ytest));
    for i=1:size(ytest,1)
       for  j=1:length(lambda)
            theta = (x'*diag(w(:,j))*x)\(x'*diag(w(:,j))*ytest(i,:)');
            ywtest(i,j)=x(j,:)*theta;
       end  
    end
    save('smoothed_y_test.mat','ywtest');
else
    load('smoothed_y_test.mat','ywtest');
end

f_left=ywtrain(:,1:find(lambda==1299));
f_right=ywtrain(:,find(lambda==1300):find(lambda==1599));
f_left_predict=zeros(size(f_left));
k=3;
for j=1:size(f_right,1)

    d_right=sum((f_right(j,:)-f_right).^2,2);
    h=max(d_right);
    [~,nk]=sort(d_right);
    nk=nk(1:k);
    ker_d_right=1-d_right(nk)./h;
    ker_d_right(find(ker_d_right<0))=0;
    f_left_predict(j,:)= (ker_d_right'*f_left(nk,:))./sum( ker_d_right);
    
end

train_error=sum((f_left-f_left_predict).^2,2);

figure;
plot(1:size(f_left,1),train_error);
avg_train_error=mean(train_error);

[~,worst]=max(train_error);
figure;
plot(lambda(1):1299,f_left(worst,:)); hold on; 
plot(lambda(1):1299,f_left_predict(worst,:));


f_left_test=ywtest(:,1:find(lambda==1299));
f_right_test=ywtest(:,find(lambda==1300):find(lambda==1599));
f_left_test_predict=zeros(size(f_left_test));
k=3;
for j=1:size(f_right_test,1)

    d_right=sum((f_right_test(j,:)-f_right).^2,2);
    h=max(d_right);
    [~,nk]=sort(d_right);
    nk=nk(1:k);
    ker_d_right=1-d_right(nk)./h;
    ker_d_right(find(ker_d_right<0))=0;
    f_left_test_predict(j,:)= (ker_d_right'*f_left(nk,:))./sum( ker_d_right);
    
end

test_error=sum((f_left_test-f_left_test_predict).^2,2);

figure;
plot(1:size(f_left_test,1),test_error);
avg_test_error=mean(test_error);

figure;
plot(lambda(1):1299,f_left(1,:)); hold on; 
plot(lambda(1):1299,f_left_predict(1,:));
figure;
plot(lambda(1):1299,f_left(6,:)); hold on; 
plot(lambda(1):1299,f_left_predict(6,:));

[~,worst_test]=max(test_error);
figure;
plot(lambda(1):1299,f_left(worst_test,:)); hold on; 
plot(lambda(1):1299,f_left_predict(worst_test,:));