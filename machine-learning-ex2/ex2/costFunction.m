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

%   term_1 =  y*log(h)  term_2 = (1-y)log(1-h)
% note here y is mx1 dimensions and h is mx1 dimensions so we have to take transpose of y for vector multiplication
% define h first

h = sigmoid(X*theta);

term_1 = (y')*log(h);
term_2 = (1-y')*log(1-h);

J = (1/m)*(-term_1 - term_2);

grad = (1/m)*X'*(h - y);			% dim of X are mxN+1 and dim of y and h are mx1 so I have taken transpose






% =============================================================

end
