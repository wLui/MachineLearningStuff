function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

% Recall the equivalent formula for multiplying X and Theta: X * Theta'
unreg_inner = (X * Theta' - Y).^2;

% Now make sure we look only at the places where R(i, j) == 1
unreg = sum(sum(unreg_inner(R == 1)))/2;
J = unreg;

% Compute unregularized gradients for Theta_grad and X_grad
unreg_inner2 = (X * Theta' - Y);
relPortion = unreg_inner2 .* R; % Extract the nonzero parts
unreg_X_grad = relPortion * Theta;
unreg_Theta_grad = relPortion' * X;

% Result is same dimensions as X
X_grad = unreg_X_grad;
% Result is same dimensions as Theta
Theta_grad = unreg_Theta_grad;

% Add regularization to cost
Theta_sq = Theta .* Theta;
X_sq = X .* X;
J = unreg + (lambda/2 * sum(sum(Theta_sq)) )  + (lambda/2 * sum(sum(X_sq)));

% Add regularization to gradients
X_grad = unreg_X_grad + lambda*X;
Theta_grad = unreg_Theta_grad + lambda*Theta;


% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
