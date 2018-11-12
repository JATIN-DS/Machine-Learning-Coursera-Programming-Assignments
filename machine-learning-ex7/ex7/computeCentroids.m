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

m = size(idx,1);
% this block of code takes O(Kn) time to run
%{		
	for i=1:K
		count = 0;
		mu = zeros(1,n);
		for j=1:m
			if idx(j) == i 						% this line means j'th example has cluster i
				count = count + 1;
				mu = mu + X(j,:);
			end
		end

		centroids(i,:) = mu./count;
	end
%}

number_of_examples_in_cluster = zeros(K,1);

	%this block if code takes O(n) to run

	for i=1:m
		cluster_number = idx(i);
		number_of_examples_in_cluster(cluster_number,1) = number_of_examples_in_cluster(cluster_number,1) + 1;
		centroids(cluster_number,:) = centroids(cluster_number,:) + X(i,:);
	end

	for i=1:K
		centroids(i,:) = centroids(i,:)./(number_of_examples_in_cluster(i));
	end




% =============================================================


end

