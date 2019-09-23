function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
% Note: grad should have the same dimensions as theta
%

% Set up cost function
% J = 1/m * sum(-y_i .* log(h_theta(x_i)) - ( 1 - y_i) * log(1 - h_theta(x_i)))
% Note - x_i is row i

% theta - is (n + 1) x 1, where we have n features
% X - is m x (n + 1), where m == number of samples

% Set up h_theta
% h_theta(x) = sigmoid(theta'*x) == sigmoid(X * theta) for our implementations
h_theta = sigmoid(X * theta);

% compute log and other temp variables
lg_h_pos = log(h_theta);
lg_h_neg = log(1 - h_theta);

y_sub = (1 - y);

% Perform computation of y .* lg_h_pos and the other part
p_pos = -1 * y .* lg_h_pos;
p_neg = y_sub .* lg_h_neg;

% finalize matrix
tot_mat = p_pos - p_neg;

% J - sum everything and divide by m
J = 1/m .* sum(sum(tot_mat));

% Compute grad
% gradient of jth theta: 1/m .* sum(h_theta(x_i) - y_i)*x_i_j
diff = h_theta - y;
for j = 1:length(theta)
	% Want to multiply difference i by x[i][j], as we're computing jth derivative
	extract = X(:,j);
	tempCol = extract .* diff;
	grad(j) = 1/m .* sum(tempCol);

end


% =============================================================

end
