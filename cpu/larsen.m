function verify = larsen(flat_mri, model_index, g)
% line nos 
X = single(flat_mri);
X(:, model_index) = [];
X = X(1: end - 1, :);
y = flat_mri(2: end, model_index);
X = zscore(X);
y = zscore(y);
[n p] = size(X);

maxVariables = min(n,p);
maxSteps = 8*maxVariables;
b = single(zeros(p, 1));
mu = single(zeros(n, 1));
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
		temp = [(c - cmax) ./ (cd - cmax); (c + cmax) ./ (cd + cmax)];
		temp = sort(temp(temp > 0));
		if isempty(temp)
			printf('Error no +ve direction\n');
			return;
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
	check = (X(:, A)' * X(:, A) * b(A) - X(:, A)' * y) ./ (a2 * sign(b(A))); % = -g
	mx = max(check);
	mn = min(check);
	ch = (mx - mn) * 200 / abs(mx + mn);
	printf("%d, %d, %f, %f, %f\n", step, size(A, 2), mx, mn, ch);
	if step > 1
			G = -((upper1 - a2) / (normb - a1));
			if G < g
				break;
			end
	end
	
	upper1 = a2;
	normb = a1;
end

% tmp = b;
% display("beta before");
% display(b(A));

% sg = sign(b(A));
% XA = X(:, A);

% Yh = y - XA * inv(XA' * XA) * XA' * y;
% Z = g * XA * inv(XA' * XA) * sg;

% p = 1 - Z' * Z;
% q = Yh' * Z + Z' * Yh; 													% Yh' * Z + Z' * Yh = 2 * Yh' * Z
% r = -Yh' * Yh;

% a21 = (-q + sqrt(q * q - 4 * p * r)) / (2 * p);
% a22 = (-q - sqrt(q * q - 4 * p * r)) / (2 * p);
% if a21 > 0 && a22 > 0
% display("controversy");
% elseif a21 > 0
% a2 = a21;
% elseif a22 > 0
% a2 = a22;
% end

% display(a2);

% b(A) = inv(XA' * XA) * (XA' * y - g * a2 * sg);

% display("beta after");
% display(b(A));

% verify = sum(sign(tmp) == sign(b(A)));
% verify = verify == size(A, 2);
% display("signs match?");
% display(verify);

% verify = sum(sign(tmp(A)) == sign(b(A)));
% verify = verify == size(A, 2);
% display("signs match?");
% display(verify);
% 306 319
end