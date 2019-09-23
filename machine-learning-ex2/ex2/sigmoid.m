function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

% Recall - sigmoid(z) = 1/(1 + exp(-z))
expPart = exp(-z);
denom = 1 + expPart;
g = 1 ./ (denom);



% =============================================================

end
