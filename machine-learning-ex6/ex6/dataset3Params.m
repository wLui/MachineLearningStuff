function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% Setup values of C and sigma to iterate over
C_vec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sig_vec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
minErr = 100;
for i = 1:length(C_vec)
	C_curr = C_vec(i);
	for j = 1:length(sig_vec)
		sigma_curr = sig_vec(j);
		% Generate the model. Remember to use training set data
		% as opposed to cross validation set
		% Also we're doing gausian kernel
		% Shouldn't we pass in sigma_curr into the function...?

		% It seems that doing @(f, g)fnctn(f, i, g) ignores params f, g
		fctnHandle = @(x1, x2)gaussianKernel(x1, x2, sigma_curr);
		model = svmTrain(X, y, C_curr, fctnHandle);

		predictions  = svmPredict(model, Xval);
		err = mean(double(predictions ~= yval));
		if(err < minErr)
			minErr = err;
			C = C_curr;
			sigma = sigma_curr;
		end
	end
end



% =========================================================================

end
