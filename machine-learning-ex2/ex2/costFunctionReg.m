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



%   term_1 =  y*log(h)  term_2 = (1-y)log(1-h)
% note here y is mx1 dimensions and h is mx1 dimensions so we have to take transpose of y for vector multiplication
% define h first

h = sigmoid(X*theta);

term_1 = (y')*log(h);
term_2 = (1-y')*log(1-h);

%for regularization we need theta0 = 0
theta(1) = 0;
J = (1/m)*(-term_1 - term_2) + (1/(2*m))*lambda*sum(theta.^2);

grad = (1/m)*X'*(h - y) + (1/m)*lambda*theta;			% dim of X are mxN+1 and dim of y and h are mx1 so I have taken transpose




% =============================================================

end
