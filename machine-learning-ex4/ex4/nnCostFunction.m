function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% Unregularized Cost function computation (Part 1)
% So for each layer l...
% Compute z_l = a_(l - 1) * Theta_(l)'
% Since we have the Theta_1 and Theta_2 unrolled already we'll just do this explicitly for 3 layers
% Add the extra bias unit only when computing the next round of activations
a_1 = [ones(m, 1), X];
z_2 = a_1 * Theta1';
a_2 = [ones(size(z_2,1), 1), sigmoid(z_2)];

z_3 = a_2 * Theta2';
a_3 = sigmoid(z_3);

% a_3 == output
% Set up Y s.t it represents the output of each as a 1 X num_labels output
Y = zeros(m, num_labels);
for(i = 1:m)
	Y(i, y(i)) = 1;
end

% Usual logistic cost function computation - J = 1/m * y * log(h_theta) - ( (1 - y)*log(1 - h_theta))
h_theta = a_3;
y_neg = (1 - Y);
lg_pos = log(h_theta);
lg_neg = log(1 - h_theta);
p1 = Y .* lg_pos;
p2 = y_neg .*lg_neg;
tot = (-1 * p1 )  - p2; 
J = 1/m * sum(sum(tot));

% Unregularized backpropagation gradient calculation (Part 2)
% Had lotsa trouble on this section for some reason :[
delt_3 = a_3 - Y;
% Note the z_2 doesn't include the bias unit
delt_2 = (delt_3 * Theta2) .* sigmoidGradient([ones(size(z_2, 1), 1), z_2]);
delt_2 = delt_2(:,2:end);
% Compute accumulation
accum_1 = delt_2' * a_1;
accum_2 = delt_3' * a_2;
Theta1_grad = 1/m * accum_1;
Theta2_grad = 1/m * accum_2;


% Regularized cost function (Part 3)
% Note - we just need to square all terms in both Theta matrices, take out the first bias term, then
% multiply the sum of these sums by lambda/2m and add to the unregularized cost
Theta1_sq = Theta1 .* Theta1;
% Note - Theta(i)(j) helps compute activation unit i and will be multiplied by term j.
% So, we need to 0 out when Theta1 is applied to bias unit (when j = 0, or 1 in Octave)
Theta1_sq(:,1) = 0;
Theta2_sq = Theta2 .* Theta2;
Theta2_sq(:,1) = 0;
sum_theta1_sq = sum(sum(Theta1_sq));
sum_theta2_sq = sum(sum(Theta2_sq));
reg_cost_term = (lambda/(2*m)) .* (sum_theta1_sq + sum_theta2_sq);
J = J + reg_cost_term;

% Regularized gradient (Part 3)
% Add regularization term - noting that we ignore the bias unit contribution
Theta1_temp = Theta1;
Theta1_temp(:,1) = 0;
Theta2_temp = Theta2;
Theta2_temp(:,1) = 0;

reg_grad1_term = lambda/m * Theta1_temp;
reg_grad2_term = lambda/m * Theta2_temp;

Theta1_grad = Theta1_grad + reg_grad1_term;
Theta2_grad = Theta2_grad + reg_grad2_term;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
