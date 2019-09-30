function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%


% UNREGULARIZED CALCULATION
% Determine cost function J - unregularized
% X: # samples x features, theta: features x 1, y: samples x 1
h_theta = sigmoid(X  * theta);
lg_pos = log(h_theta);
lg_neg = log(1 - h_theta);
y_neg = (1 - y);
tot = (-1 * y).*lg_pos - (y_neg .* lg_neg);
J = 1/m * sum(tot);

% Compute gradient - unregularized
% Recall - jth partial = 1/m * sum( (h_theta - y) * x_j, i = 1 to m )
diff = h_theta - y;
grad = 1/m * X' * diff;


% REGULARIZED CALCULATION
% Regularize cost
theta_sq = theta .* theta;
theta_sq(1) = 0;
reg_cost_term = lambda/(2*m) * sum(theta_sq);
J = J + reg_cost_term;

% Regularize gradient by adding lambda/m * theta_j
theta_temp = theta;
theta_temp(1) = 0;
reg_grad_term = lambda/(m) * theta_temp;
grad = grad + reg_grad_term;


% =============================================================

grad = grad(:);

end
