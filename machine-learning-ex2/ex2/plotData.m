function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
% dim of X are mx(n+1) and dim of y are mx1
m = size(y,1);

for c=1:m
	if y(c)==1
		plot(X(c,1),X(c,2),'k+')
	else
		plot(X(c,1),X(c,2),'ko')
	end
	hold on
end









% =========================================================================



hold off;

end
