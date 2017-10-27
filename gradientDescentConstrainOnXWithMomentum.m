function [ x_est, E ] = gradientDescentConstrainOnXWithMomentum(x0, x, A, y, lambda, ...
                             T, stepsize, gamma, is_zero, is_all_iterations)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% x_est = argmin ||Ax-y||^2 + lambda ||x||^2 
%
% PARAMS:
%
% - x0 : side information;  
% - x  : original signal;
% - A  : transform matrix;
% - y  : qunatised-sparsified representation of x;
% - lambda : Lagrangian multiplier for side information;
% - T  : number of iterations;
% - stepsize : start step size of descent;
% - gamma    : decresing speed of step size;  
% - is_zero  : type of start initialisation: 0 - start from prior; 1 - strart from zero
% - is_all_iterations :  1 - finish after T iteration; 0 - finish after convergence;
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% --- Functions -----------------------------------------------------------
F = @(z, A, y) 0.5*(norm(A*z - y)^2+lambda*norm(z)^2);
% -------------------------------------------------------------------------

E     = [];  % estimated error 
iter  = 0;   % number of iterations
alpha = 0.1; % momentum weight 

n      = size(A, 2);
B      = A'*A + lambda*eye(n);
C      = A'*y;

x_kp1 = zeros(size(x));
if is_zero
    x_k   = zeros(size(x));
    x_km1 = zeros(size(x));
else
    x_k   = x0;  
    x_km1 = x0;
end

while (is_all_iterations && iter < T) || ...
        (is_all_iterations  == 0 && (iter == 0 || (iter < T && norm(x-x_k) > norm(x-x_kp1))))
    if iter >= 2
       x_km1 = x_k;
    end
    if iter ~= 0
        x_k = x_kp1;
    end
    iter = iter + 1;
        
    gradvalue = B*x_k - C;
    numTrials = 1;
    newStep   = stepsize;    
    
    x_kp1 = x_k - newStep*gradvalue + alpha*(x_k-x_km1);
    difference  = F(x_kp1,A,y) - F(x_k,A,y) ...
                - gradvalue'*(x_kp1-x_k) - (1/(2*newStep))*norm(x_kp1-x_k)^2; 
    % backtracking line search
    while numTrials < 500 && difference > 0,
        numTrials   = numTrials + 1;
        newStep     = newStep*gamma;
        x_kp1       = x_k - newStep*gradvalue + alpha*(x_k-x_km1);
        difference  = F(x_kp1,A,y) - F(x_k,A,y) ...
            - gradvalue'*(x_kp1-x_k) - (1/(2*newStep))*norm(x_kp1-x_k)^2;
    end

    E(iter) = norm(x-x_kp1);
end

if is_all_iterations == 0 && length(E) > 1
    E(end) = [];
end

x_est = x_k;

end

