function [ x_est, E ] = AMPConstrainOnX(x0, x, A, y, lambda, ...
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

n      = size(A, 2);
B      = A'*A + lambda*eye(n);
C      = A'*y;

v      = zeros(size(y));
x_cur  = zeros(size(x));
if is_zero
    x_prev = zeros(size(x));
else
    x_prev  = x0;    
end

while (is_all_iterations && iter < T) || ...
        (is_all_iterations  == 0 && (iter == 0 || (iter < T && norm(x-x_prev) > norm(x-x_cur))))
    if iter ~= 0
        x_prev = x_cur;
    end
    iter = iter + 1;
    
    b = norm(x_prev)/n;
    v = y - A*x_prev + b*v;        
        
    gradvalue = B*x_prev - C;
    numTrials = 1;
    newStep   = stepsize;    
    
    x_cur       = x_prev - newStep*gradvalue + A'*b*v;
    Fprev       = F(x_prev,A,y);
    difference  = F(x_cur,A,y) - Fprev ...
                - gradvalue'*(x_cur-x_prev) - (1/(2*newStep))*norm(x_cur-x_prev)^2; 
            
    % backtracking line search        
    while numTrials < 500 && difference > 0,
        numTrials   = numTrials + 1;
        newStep     = newStep*gamma;
        x_cur       = x_prev - newStep*gradvalue + A'*b*v;
        difference  = F(x_cur,A,y) - Fprev ...
            - gradvalue'*(x_cur-x_prev) - (1/(2*newStep))*norm(x_cur-x_prev)^2;
    end

    E(iter) = norm(x-x_cur);
end

if is_all_iterations == 0 &&  length(E) > 1
    E(end) = [];
end

x_est = x_prev;

end

