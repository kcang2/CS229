%% importing, normalizing and splitting data
% rng(20171115, 'twister');
% if (exist('train_data', 'var') ~= 1)
%     train_data = load('images_train.csv');
%     train_label = load('labels_train.csv');
%     test_data = load('images_test.csv');
%     test_label = load('labels_test.csv');
% end
% 
% n = size(train_data, 2);
% c = 10;
% train_size = 50000;
% p = randperm(size(train_data, 1));
% 
% % converting labels to one hot encoding
% y_train = zeros(c, size(train_label, 1));
% y_train(train_label + 1 + c * (0:size(train_label, 1)-1)') = 1;
% y_train = y_train';
% 
% y_test = zeros(c, size(test_label, 1));
% y_test(test_label + 1 + c * (0:size(test_label, 1)-1)') = 1;
% y_test = y_test';
% 
% X_train_all = train_data(p, :);
% y_train_all = y_train(p, :);
% X_train = X_train_all(1:train_size, :);
% X_dev = X_train_all(train_size+1:end, :);
% y_train = y_train_all(1:train_size, :);
% y_dev = y_train_all(train_size+1:end, :);
% X_test = test_data;
% 
% %normalize the data
% avg = mean(mean(X_train));
% s = std(reshape(X_train, [], 1));
% 
% X_train = (X_train - avg) / s;
% X_dev = (X_dev - avg) / s;
% X_test = (X_test - avg) / s;

%% training the network (without regularization)
% m = train_size;
% h1 = 300; % 300 units in hidden layer
% 
% % initialize the parameters
% W1 = randn(h1, n);
% b1 = zeros(h1, 1);
% W2 = randn(c, h1);
% b2 = zeros(c, 1);
% 
% num_epoch = 30;
% batch_size = 1000;
% num_batch = m / batch_size;
% learning_rate = 5;
% lambda = 0.0001;
% train_loss = zeros(num_epoch, 1);
% dev_loss = zeros(num_epoch, 1);
% train_accuracy = zeros(num_epoch, 1);
% dev_accuracy = zeros(num_epoch, 1);

%% Your code here

% for i=1:num_epoch
%     for j=1:num_batch
%         eg = ((j-1)*batch_size + 1):(j*batch_size);
%         [dW1, db1, dW2, db2] = backward_prop(X_train(eg,:),...
%             y_train(eg,:), W1, b1, W2, b2, lambda); %% CHANGE LAMBDA HERE!
%         W1 = W1 - learning_rate*dW1;
%         W2 = W2 - learning_rate*dW2;
%         b1 = b1 - learning_rate*db1;
%         b2 = b2 - learning_rate*db2;
%     end
%      [h_train, ~, train_loss(i)] = forward_prop(X_train, y_train, W1, b1, W2, b2);
%      [h_dev, ~, dev_loss(i)] = forward_prop(X_dev, y_dev, W1, b1, W2, b2);
%      [r_train, ~] = find(y_train');
%      [r_dev, ~] = find(y_dev');
%      train_accuracy(i) = sum(h_train == r_train)/m;
%      dev_accuracy(i) = sum(h_dev == r_dev)/length(y_dev);
% end

% Your code end here
%% plotting and displaying results

% figure(1);
% plot(1:num_epoch, train_loss, 'r', ...
%     1:num_epoch, dev_loss, 'b');
% legend('training set', 'dev set');
% xlabel('epochs');
% ylabel('loss');
% 
% figure(2);
% plot(1:num_epoch, train_accuracy, 'r', ...
%     1:num_epoch, dev_accuracy, 'b');
% legend('training set', 'dev set');
% xlabel('epochs');
% ylabel('accuracy');

% save('W1_no_reg','W1'); save('W2_no_reg','W2'); save('b1_no_reg','b1'); ...
%     save('b2_no_reg','b2'); 

% save('W1_reg','W1'); save('W2_reg','W2'); save('b1_reg','b1'); ...
%     save('b2_reg','b2'); 

[~, test_pred, ~] = forward_prop(X_test, y_test, W1, b1, W2, b2);
[~, y_c_test_pred] = max(test_pred, [], 2);
[~, y_c_test] = max(y_test, [], 2);
test_accuracy = sum(y_c_test_pred == y_c_test) / size(y_test, 1);
fprintf('test set accuracy: %f \n', test_accuracy);
