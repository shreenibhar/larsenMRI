% Deletes a variable from the X'X matrix in a Cholesky factorisation R'R = 
% X'X. Returns the downdated R. This function is just a stripped version of 
% Matlab's qrdelete. 
function R = choldelete_gpu(R,j) 
R(:,j) = []; % remove column j 
n = size(R,2); 
for k = j:n 
  p = k:k+1; 
  [G,R(p,k)] = planerot(R(p,k)); % remove extra element in column 
  if k < n 
    R(p,k+1:n) = G*R(p,k+1:n); % adjust rest of row 
  end 
end 
R(end,:) = []; % remove zero'ed out row 