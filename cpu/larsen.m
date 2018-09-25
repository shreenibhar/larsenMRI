% Change Log
% L1 norm(b) changed everywhere to L1 norm(b(A))
% Implemented correction equations

function [b, l1_out, err_out, g_out, step] = larsen(flat_mri, model_index, max_l1, min_l2, g, max_ss, max_steps)

X = double(flat_mri);
X(:, model_index) = [];
X = X(1: end - 1, :);
y = flat_mri(2: end, model_index);
X = zscore(X) / sqrt(size(X, 1) - 1);
y = zscore(y) / sqrt(size(y, 1) - 1);
[n p] = size(X);

if max_l1 <= 0 max_l1 = 1000; end;
if (g <= 0) g = 0.001; end;
if (max_ss <= 0) max_ss = n; end;
maxVariables = min([n  p max_ss]);
if (max_steps <= 0) max_steps = 8 * maxVariables; end;
max_steps = min(8 * maxVariables, max_steps);

err = 1;
l1 = 0;
b = double(zeros(p, 1));
mu = double(zeros(n, 1));
I = 1:p;
A = [];
lassoCond = 0;
step = 0;
G = g + 1;

while (length(A) < maxVariables) && (step < max_steps) && (G > g) && (err > min_l2) && (l1 < max_l1)

    if lassoCond
		I = [I A(dropIdx)];
		A(dropIdx) = [];
	end

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

	gamma_tilde = b(A) ./ (b(A) - b_OLS);
	gamma_tilde(gamma_tilde <= 0) = inf;
	[gamma_tilde dropIdx] = min(gamma_tilde);

	if isempty(I)
		gamma = 1;
	else
		cd = X'*d;
		temp = [(c(I) - cmax) ./ (cd(I) - cmax); (c(I) + cmax) ./ (cd(I) + cmax)];
		[temp2 inds2] = sort(temp(temp > 0));
		if isempty(temp2)
			printf('Error no +ve direction\n');
			return;
		end
		gamma = temp2(1);
	end

	if gamma_tilde < gamma
		lassoCond = 1;
		gamma = gamma_tilde;
    end

    b_prev = b;
	b(A) = b(A) + gamma*(b_OLS - b(A));
    
    mu = mu + gamma * d;
    step = step + 1;
    l1 = norm(b(A), 1);
    err = norm(y - mu, 2);

    G_array = abs(X(:,A)' * (y - mu) / err);
    if ((max(G_array) - min(G_array)) / max(G_array) > 1e-5)
        disp(sprintf('Warning: Active sets do not seem to have same derivatives (possible numerical unstability)\nMax_g min_g %g %g %g\n', max(G_array), min(G_array), (max(G_array)-min(G_array))/max(G_array)));
    end
    G = min(G_array);
end

XA = X(:, A);
sb = sign(b(A));
if lassoCond
	sb(dropIdx) = sign(b_prev(A(dropIdx)));
end

if (l1 > max_l1)
    disp(sprintf('Applying L1 bound correction: L1 %g max_l1 %g\n', l1, max_l1));
    l1 = norm(b_prev(A), 1);
    delta = sum(sign(b_prev(A)) .* (b_OLS - b_prev(A)));
    gamma = (max_l1 - l1) / delta;
    b(A) = b_prev(A) + gamma * (b_OLS - b_prev(A));
    l1 = max_l1;
    mu = X(:, A) * b(A);
    err = norm(y - mu, 2);
    G_array = abs(X(:,A)' * (y - mu) / err);
    G = min(G_array);
end
if (err < min_l2)
    disp(sprintf('Applying error bound correction: quadratic equation here. Err %g min_l2 %g\n', err, min_l2));
    yhyh = y' * (y - XA * inv(XA' * XA) * XA' * y);
	zz = sb' * inv(XA' * XA) * sb;
	G = sqrt((min_l2 * min_l2 - yhyh) / (min_l2 * min_l2 * zz));
	err = min_l2;
	b(A) = inv(XA' * XA) * (XA' * y - G * err * sb);
	l1 = norm(b(A), 1);
end
if (G < g)
    disp(sprintf('Applying G  correction:G %g g %g\n', G, g));
    yhyh = y' * (y - XA * inv(XA' * XA) * XA' * y);
    zz = sb' * inv(XA' * XA) * sb;
	err = sqrt(yhyh / (1 - g * g * zz));
	G = g;
	b(A) = inv(XA' * XA) * (XA' * y - G * err * sb);
	l1 = norm(b(A), 1);
end

verify = sum(sign(b(A)) == sb);
if verify ~= size(A, 2)
	display([sb sign(b(A))]);
	disp(sprintf('Signs dont match!\n'));
end

l1_out = l1;
err_out = err;
g_out = G;

end
