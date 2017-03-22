%% Fast Cholesky insert and remove functions 
% Updates R in a Cholesky factorization R'R = X'X of a data matrix X. R is 
% the current R matrix to be updated. x is a column vector representing the 
% variable to be added and X is the data matrix containing the currently 
% active variables (not including x). 
function R = cholinsert_gpu(R, x, X, lambda) 
diag_k = (x'*x + lambda)/(1 + lambda); % diagonal element k in X'X matrix 
if isempty(R) 
  R = sqrt(diag_k); 
else 
  col_k = x'*X/(1 + lambda); % elements of column k in X'X matrix 
  R_k = R'\col_k'; % R'R_k = (X'X)_k, solve for R_k 
  R_kk = sqrt(diag_k - R_k'*R_k); % norm(x'x) = norm(R'*R), find last element by exclusion 
  R = [R R_k; [zeros(1,size(R,2)) R_kk]]; % update R 
end 