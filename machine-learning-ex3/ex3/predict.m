function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Remember - X is m x n, Theta is act units x (prev features + 1)
% So to compute inputs to sigmoid, we need to transpose Theta at each step
% Basically, we need to compute the final result. We know that there is one hidden layer.

% Insert bias for computation of z_2, input to sigmoid to hidden layer 
a_1 = X;
z_2 = [ones(size(a_1), 1), a_1] * Theta1';
a_2 = sigmoid(z_2);

% Compute the final output, with the activation units of hidden layer
z_3 = [ones(size(a_2), 1), a_2] * Theta2';
a_3 = sigmoid(z_3);

% Compute the predicted result- (look columnwise for each row, pick largest column)
[pred_max, ind_max] = max(a_3, [], 2);

p = ind_max;





% =========================================================================


end
