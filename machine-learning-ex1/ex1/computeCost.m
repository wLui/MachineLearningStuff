function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% Remember - m training samples, n features 
% so X is m x (n + 1), as we add an extra column for theta_0 == 1
% Theta is (n + 1) x 1
% Hypothesis: h_theta(x) = theta'*X
% NOTE - in programming assignments it is expected that hyp = X * theta
% J(theta) = 1/2m * sum( (h_theta(x_i) - y_i) ^ 2, 1, m)
hyp = X * theta;
sq_diff = (hyp - y).^2;
J = 1/(2*m) * sum(sum(sq_diff));




% =========================================================================

end
