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

%%% Load from ex6data3: 
%%% You will have X, y in your environment
%%load('ex6data3.mat');

min_error = 1;
C = [0.01,0.03,0.1,0.3,1,3,10,30];
sigma = [0.01,0.03,0.1,0.3,1,3,10,30];

for i = 1:8
    for j = 1:8
        
        % Train the SVM
        model= svmTrain(X, y, C(i), @(x1, x2) gaussianKernel(x1, x2, sigma(j)));
        
        %predict
        predy = svmPredict(model,Xval);
        
        %error rate
        error = mean(double(predy~= yval));
        
        if error < min_error
            best_C = C(i);
            best_sigma = sigma(j);
            min_error = error;
        end
    end
end

C = best_C;
sigma = best_sigma;
%%fprintf('best_C is %f,best_sigma is %f.\n',best_C,best_sigma); 







% =========================================================================

end
