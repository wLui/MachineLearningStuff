function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
alpha_m = alpha/m;

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
    % Recall Gradient Descent formula
    % Assign to theta_j: theta_j - alpha/m * sum( (hyp(x_samp_i) - y_i)x_samp_i_feat_j)
    hyp_curr = X * theta;
    diff = (hyp_curr - y); % Diff is 97 x 1 => samples
    
    % want to mult diff elementwise by samples in theta(i)
    % Would like to vectorize this, if possible
    for j = 1:length(theta)
      inner = diff .* X(:,j);
      theta(j) = theta(j) - alpha_m * sum(inner); 
    end
    % Vectorizes but doesn't perform simulatenous update.
    #{
    inner = X' * (X*theta - y);
    theta = theta - (alpha_m * inner);
   #}
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
end

end
