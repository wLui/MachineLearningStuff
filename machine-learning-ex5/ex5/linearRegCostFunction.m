function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% Compute regularized linear regression
% Remember - X is m x n, where n == number of features (including bias)
% Theta is n x 1, where n == number of features

% Compute inner summation for l.r
h_theta = X * theta;
diff = h_theta - y;
diff_sq = diff .* diff;

% Compute reg term
theta_sq = theta .* theta;
% Zero out the bias feature 
theta_sq(1) = 0;

J = 1/(2 * m) * sum(diff_sq) +  ( lambda/(2 * m) * sum(theta_sq) );

% Now compute gradient
% Note it may be vectorized
% Unreg formula: jth partial = 1/m * sum(h_theta(x_i) - y_i)*xj_i, i = 1 to m)
% Stil need to verify by hand why this works...
grad = 1/m * X' * diff;

theta_temp = theta;
theta_temp(1) = 0;
grad = grad + ( lambda/m * theta_temp);








% =========================================================================

grad = grad(:);

end
