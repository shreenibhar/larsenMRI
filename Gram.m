function G = Gram(X, A)
	G = X(:, A)' * X(:, A);
end
