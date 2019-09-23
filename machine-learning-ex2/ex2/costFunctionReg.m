function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
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

% Repeating what was done earlier in costFunction.m
h_theta = sigmoid(X * theta);

lg_pos = log(h_theta);
lg_neg = log(1 - h_theta);
y_sub = 1 - y;

p_pos = -1 * y .* lg_pos;
p_neg = y_sub .* lg_neg;

sum1 = 1/m .* sum((p_pos - p_neg));

% Deal with regularization part
theta_sq = theta .* theta;

% Don't worry about theta_0
theta_sq(1) = 0;
sum2 = lambda/(2*m) .* sum(theta_sq);

J = sum1 + sum2;

% Do gradient descent
diff = h_theta - y;
for(j = 1:length(theta))
	extract = X(:, j);
	tempCol = extract .*diff;
	grad(j) = 1/m .* sum(tempCol);
	if(j != 1)
		grad(j) = grad(j) + ( (lambda/m) * theta(j) );
	end
end

% =============================================================

end
