function [b, steps, G, a2, drop] = larsen(flat_mri, model_index, g)
X = flat_mri;
X(:, model_index) = [];
X = X(1: end - 1, :);
y = flat_mri(2: end, model_index);
X = zscore(X);
y = zscore(y);
[n p] = size(X);
X /= norm(X(:, 2), 2);
y /= norm(y, 2);

maxVariables = min(n,p);
maxSteps = 8*maxVariables;
b = zeros(p, 1);
mu = zeros(n, 1);
I = 1:p;
A = [];
lassoCond = 0;
step = 0;
deltar = 0;
deltax = 0;
G = 0;

while length(A) < maxVariables && step < maxSteps
	r = y - mu;
	c = X' * r;
	[cmax cidxI] = max(abs(c(I)));
	cidx = I(cidxI);
	
	if ~lassoCond 
		A = [A cidx];
		I(cidxI) = [];
	else
		lassoCond = 0;
	end

	Gram = X(:, A)' * X(:, A);
	b_OLS = Gram \ (X(: , A)' * y);
	d = X(: , A) * b_OLS - mu;

	gamma_tilde = b(A(1: end - 1)) ./ (b(A(1: end - 1)) - b_OLS(1: end - 1, 1));
	gamma_tilde(gamma_tilde <= 0) = inf;
	[gamma_tilde dropIdx] = min(gamma_tilde);

	if isempty(I)
		gamma = 1;
	else
		cd = X'*d;
		temp = [(c(I) - cmax) ./ (cd(I) - cmax); (c(I) + cmax) ./ (cd(I) + cmax)];
		temp = sort(temp(temp > 0));
		if isempty(temp)
			printf('Error no +ve direction\n');
			return
		end
		gamma = temp(1);
	end

	if gamma_tilde < gamma
		lassoCond = 1;
		gamma = gamma_tilde;
	end
		
	b(A) = b(A) + gamma*(b_OLS - b(A));
	mu = mu + gamma * d;
	
	step = step + 1;

	if lassoCond
		I = [I A(dropIdx)];
		A(dropIdx) = [];
	end

	a1 = norm(b, 1);
	a2 = norm((y - mu), 2);
	if step > 1
			G = -((upper1 - a2) / (normb - a1));
			if G < g
				break
			end
	end
	
	upper1 = a2;
	normb = a1;
end

drop = length(A);

upper1
normb
step
drop

end