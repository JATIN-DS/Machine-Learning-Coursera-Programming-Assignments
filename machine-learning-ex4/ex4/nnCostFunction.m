function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
k = num_labels; % num_labels        
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

Y = zeros(m,k);

for w=1:m 						 % this creates Y which has dimensions= no. of ex x no. of labels
	p = y(w);
	Y(w,p) = 1;
end


% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%




%   term_1 =  y*log(h)  term_2 = (1-y)log(1-h)
% note here y is mx1 dimensions and h is mx1 dimensions so we have to take transpose of y for vector multiplication
% define h first

% starting part-1

	h1 = sigmoid([ones(m, 1) X] * Theta1');
	h2 = sigmoid([ones(m, 1) h1] * Theta2');
	[dummy, p] = max(h2, [], 2);
		
	% note about h2: it has dimensions = no. of examples x no. of labels = each value lies between 0 and 1 as...
	% it is obtained from sigmoid function. Now in cost function J we need sigmoid value of final hypothesis ..
	% which is h2 in dimensions= no. of labels x 1	
for c=1:m

	actual_output = Y(c,:);
	actual_output = actual_output';		% now actual_output is a col vector of size k
	
	h = h2(c,:);		
	h = h';				%dimensions = num_labels x 1


	term_1 = (actual_output')*log(h);
	term_2 = (1-actual_output')*log(1-h);

	temp_theta1 = Theta1;
	temp_theta2 = Theta2;
	temp_theta1(:,1) = 0;			% note: first col of theta1 and Theta2 belongs to bias unit
	temp_theta2(:,1) = 0;

	theta_square = sum(sum(temp_theta1.^2)) + sum(sum(temp_theta2.^2));  % sum of all elements in matrix

	J_for_this_ex = (1/m)*(-term_1 - term_2);
	J = J + J_for_this_ex;
end
	
	J = J + (1/(2*m))*lambda*theta_square;  % now adding total cost due to regularization term
% Part -1 done here

% starting part-2

for t=1:m

	a1 = [1 X(t,:)];
	z2 = a1*Theta1';
	a2 = sigmoid(z2);
	a2 = [1 a2];
	z3 = a2*Theta2';
	a3 = sigmoid(z3);

	actual_output = Y(t,:);
	actual_output = actual_output';		% now actual_output is a col vector of size k
	delta_3 = a3' - actual_output; 			

	% for layer 2
	Theta2_trans = Theta2' ;
	abc = Theta2'*delta_3;
	abc2 = a2.*(1-a2);			% note here we want a2, padded with 1 so we cant directly use sigmoidGradient(z2)
	delta_2 = abc.*abc2';		%%I have written correctly, given wrong in their pdf
	delta_2 = delta_2(2:end,:);		

	Theta1_grad = Theta1_grad + delta_2*(a1);	%I have written correctly, given wrong in their pdf
	Theta2_grad = Theta2_grad + delta_3*(a2);

end


	Theta1_grad = (1/m)*Theta1_grad;
	Theta2_grad = (1/m)*Theta2_grad;

% PART-2 ENDS HERE

% starting PART-3

	Theta1_grad = Theta1_grad + (lambda/m)*temp_theta1;
	Theta2_grad = Theta2_grad + (lambda/m)*temp_theta2;
% part-3 ends here

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
