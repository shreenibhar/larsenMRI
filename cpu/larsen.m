function [b,steps,G,a2,error,drop] = larsen(X, y, delta, g)
%LARSEN The LARS-EN algorithm for estimating Elastic Net solutions.
%
%   BETA = LARSEN(X, Y, DELTA, STOP, GRAM, STOREPATH, VERBOSE) evaluates
%   the LARS-EN algmorithm [1] using the variables in X to approximate the
%   response y given regularization parameters DELTA and STOP (LAMBDA).
%   See the function ELASTICNET for a description of parameters X, Y,
%   DELTA, STOP, STOREPATH and VERBOSE. GRAM represents an optional
%   precomputed Gram matrix of size p by p (X is n by p). The number of
%   iterations performed is returned as a second output.
%   
%   Note: The main purpose of this function is to act as an inner function
%   for the more user-friendly functions ELASTICNET and LASSO. Direct use
%   of this function requires good understanding of the algorithm and its
%   implementation.
%
%   The algorithm is a variant of the LARS-EN algorithm [1].
%
%   References
%   -------
%   [1] H. Zou and T. Hastie. Regularization and variable selection via the
%   elastic net. J. Royal Stat. Soc. B. 67(2):301-320, 2005. 
%

%% algorithm setup
[n p] = size(X);
% Determine maximum number of active variables
if delta < eps
  maxVariables = min(n,p); %LASSO
else
  maxVariables = p; % Elastic net
end

maxSteps = 8*maxVariables; % Maximum number of algorithm steps

% set up the LASSO coefficient vector
b = zeros(p, 1);
b_prev = b;

% current "position" as LARS travels towards lsq solution
mu = zeros(n, 1);


I = 1:p; % inactive set
A = []; % active set


lassoCond = 0; % LASSO condition boolean
stopCond = 0; % Early stopping condition boolean
step = 1; % step count
deltar = 0; % change in residual
deltax = 0; % change in norm x
G = 0;

%% LARS main loop
% while not at OLS solution, early stopping criterion is met, or too many
% steps have passed 
while length(A) < maxVariables && ~stopCond && step < maxSteps
  r = y - mu;

  % find max correlation
  c = X'*r;
  [cmax cidxI] = max(abs(c(I)));
  cidx = I(cidxI); % index of next active variable
  
  if ~lassoCond 
    % add variable
    A = [A cidx]; % add to active set
    I(cidxI) = []; % ...and drop from inactive set
  else
    % if a variable has been dropped, do one step with this
    % configuration (don't add new one right away) 
    lassoCond = 0;
  end

  % partial OLS solution and direction from current position to the OLS
  % solution of X_A
  b_OLS = Gram(X,A)\(X(:,A)'*y); % same as X(:,A)\y, but faster
  d = X(:,A)*b_OLS - mu;
  
  % compute length of walk along equiangular direction
  gamma_tilde = b(A(1:end-1))./(b(A(1:end-1)) - b_OLS(1:end-1,1));
  gamma_tilde(gamma_tilde <= 0) = inf;
  [gamma_tilde dropIdx] = min(gamma_tilde);

  if isempty(I)
    % if all variables active, go all the way to the OLS solution
    gamma = 1;
  else
    cd = X'*d;
    temp = [ (c(I) - cmax)./(cd(I) - cmax); (c(I) + cmax)./(cd(I) + cmax) ];
    temp = sort(temp(temp > 0)); % faster than min(temp(temp > 0)) (!)
    if isempty(temp)
        drop = length(A);

        % return number of iterations
        steps = step - 1;
        
        return
    end
    gamma = temp(1);
  end
  
  % check if variable should be dropped
  if gamma_tilde < gamma
    lassoCond = 1;
    gamma = gamma_tilde;
  end
    
  % update beta
  b_prev = b;
  b(A) = b(A) + gamma*(b_OLS - b(A)); % update beta

  % update position
  mu = mu + gamma*d;
  
  % increment step counter
  step = step + 1;
  display(step);

  
  % If LASSO condition satisfied, drop variable from active set
  if lassoCond
    I = [I A(dropIdx)]; % add dropped variable to inactive set
    A(dropIdx) = []; % ...and remove from active set
  end
  
  Y = y;
  Y_new = X*b;
  a1 = norm(b,1);
  a2 = norm((Y-Y_new),2);
  if step > 2
      G = -((upper1-a2)/(normb- a1));
      error = a2/norm(Y_new,2);
      if G < g
        drop = length(A);

        % return number of iterations
        steps = step - 1;
        
        return
        
      end
  end
    
    upper1 = a2;

    normb = a1;

  
end

drop = length(A);

% return number of iterations
steps = step - 1;
