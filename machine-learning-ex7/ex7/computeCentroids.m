function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%


% Go through every centroid. Compute mean of all points that correspond to
% this specific centroid.

for i = 1:K
	% So we want only the indices that correspond to centroid i
	c_i = idx == i;

	% So total number of such entries is n_i
	n_i = sum(c_i);

	% Now we need to compute the sum of the corresponding x's
	nz_ind = find(c_i);
	x_i = X(nz_ind, :);

	% Now compute average
	centroids(i, :) = sum(x_i) / n_i;

end



% =============================================================


end

