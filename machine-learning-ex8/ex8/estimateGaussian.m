function [mu sigma2] = estimateGaussian(X)
%ESTIMATEGAUSSIAN This function estimates the parameters of a 
%Gaussian distribution using the data in X
%   [mu sigma2] = estimateGaussian(X), 
%   The input X is the dataset with each n-dimensional data point in one row
%   The output is an n-dimensional vector mu, the mean of the data set
%   and the variances sigma^2, an n x 1 vector
% 

% Useful variables
[m, n] = size(X);

% You should return these values correctly
mu = zeros(n, 1);
sigma2 = zeros(n, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the mean of the data and the variances
%               In particular, mu(i) should contain the mean of
%               the data for the i-th feature and sigma2(i)
%               should contain variance of the i-th feature.
%

% Note - X is m x n, where m = # samples, n = # features.
% So the mean we wanna compute is for a single feature among all samples (sum over columns)
colSum = sum(X);
mu = colSum/m;

% sigma2 is the variances of each feature. Since we have the sum of each column computed
% as a row vector (colSum variable) just take sq diff and divide by m
% The formula is sum(sqDiff(sample_feature - corresponding_mean))/m
% Need to make muMatrix to be m x n, where muMatrix[i, j] = mu[j]

% Originally tried to do sum( (colSum - mu).^2 )/m;
muMat = repmat(mu, m, 1);
varSqDiff = (X - muMat).^2;
sigma2 = sum(varSqDiff)/m;






% =============================================================


end
