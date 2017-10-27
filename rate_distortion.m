%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Rate-distortion
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
setenv('LC_ALL','C');

warning off all;

close all
clear
clc

addpath(genpath('libs'));


% --- CONSTANTS -----------------------------------------------------------
n  = 500; % number of measurement
M  = 250; % number of samples
st = ceil(M/5); % sparsity changing step
S  = [1:st:M]; % sparsity
t  = length(S);

m  = 10000; 

% gradient descent params
T        = 500;   % number of iterations
is_all_iterations = 0; % 1 - finish after T iteration; 0 - finish after convergence;
stepsize = 1;     % step size
gamma    = 0.005; % backtracking line search parameter
is_zero  = 0;     % start initialisation type: 0 - start from prior; 1 - strart from zero 


Run = 1000; % number of runs

sigma = 1; % noise standard deviation


% --- Functions -----------------------------------------------------------
thresholds  = @(sortedData, K) sortedData(K, :);
kLagestMask = @(P, K) bsxfun(@ge,P, thresholds(sort(P, 'descend'), K));
D           = @(sx, R) sx*2^(-2*R); % rate-distortion 

%% --- Prepare data -------------------------------------------------------
X_dbase = randn(m,n); 
X_dbase = X_dbase - repmat(mean(X_dbase),m,1);
X       = X_dbase;

%% --- Process ------------------------------------------------------------
D_upper_bound = zeros(Run, t);

D_map  = zeros(Run, t);
D_pinv = zeros(Run, t);

D_gd_const_x_init_x0  = zeros(Run, t);
D_gd_const_x0_init_x0 = zeros(Run, t);
D_gd_const_x0_init_0  = zeros(Run, t);
D_gd_no_priors        = zeros(Run, t);

D_amp_const_x_init_x0  = zeros(Run, t);
D_amp_const_x0_init_x0 = zeros(Run, t);
D_amp_const_x0_init_0  = zeros(Run, t);
D_amp_no_priors        = zeros(Run, t);

R = zeros(1, t);
% -------------------------------------------------------------------------
for r=1:Run
    i = 0;
    
    A    = randn(M, n); % transform matrix
    if M < n
        A = orth(A')';
    else
        A = orth(A);
    end

    x    = randn(n, 1); % original signal
    x    = x - mean(x);
    y    = A*x;   

    x0 = x + randn(n, 1)*sigma; % side infromation

    for s = S
        i = i + 1;

        R(i) = (log2(nchoosek(M,s)) + s)/n; % rate
        DS(i) = var(x)*2^(-2*R(i)); % distortion
        
        ys   = sign(y).*kLagestMask(abs(y), s); % quantized-sparsified representation of x
        
        % --- estimation of normalized coefficient for pseudo-inverse solution ----
        [U, Sigm, V] = svd(A);
        Ainv = V*pinv(Sigm)*U';
        fX = X*A';
        for ii=1:m
            fX(ii, :) = sign(fX(ii, :)').*kLagestMask(abs(fX(ii, :)'), s); 
        end
        X_hat  = fX*(Ainv)';
        b_ps   = trace(X'*X_hat)/trace(X_hat'*X_hat); % normalized coefficient for pseudo-inverse solution  
        %--------------------------------------

        % --- 1. Upper bound ----
        lambda              = var(y-ys)/var(x-x0); % Lagrangian multiplier 
        D_upper_bound(r, i) = rank(A)*norm(  A'*(A*x-ys) + lambda*(x-x0)  )^2/(n*(1+lambda));
        
        % --- 2. MAP estimation ----
        x_est       = x0*var(x)/(var(x)+sigma^2);
        D_map(r, i) = norm(x-x_est)^2/n;

        % --- 3. pseudo-inverse ----
        x_est        = Ainv*ys;
        D_pinv(r, i) = norm(x-x_est*b_ps)^2/n;
        
        % --- 4. multi-layer gradient descent ----

        % ----- 4.1 ||Ax-y||^2 + lambda ||x||^2 init via x0
        x_prior = x0; % side informaiton
        j       = 1;  % number of layers 
        D_j     = 0; % obtained distortion at eat layer
        D_all   = 0; % obtained distortion at all layers
        x_opt   = zeros(size(x)); % optimal estimation
        while j <= 2 || min(D_j) < min(D_all)
            if j > 1
                D_all   = cat(2, D_all, D_j);
                x_prior = x_est*var(x)/(var(x)+var(x-x_est));
            end  
            lambda = var(y-ys)/var(x);
            [x_est, D_j] = gradientDescentConstrainOnXWithMomentum(x_prior, x, A, ys, lambda, T, stepsize, ...
                gamma, 0, is_all_iterations);
            if norm(x-x_est) < norm(x-x_opt)
                x_opt = x_est;
            end
            j = j + 1;
        end
        D_gd_const_x_init_x0(r, i) = norm(x-x_opt)^2/n;        
     
        % ----- 4.2 ||Ax-y||^2 + lambda ||x-x0||^2 init via x0
        x_prior = x0; % side informaiton
        j       = 1;  % number of layers 
        D_j     = 0; % obtained distortion at eat layer
        D_all   = 0; % obtained distortion at all layers
        x_opt   = zeros(size(x)); % optimal estimation
        while j <= 2 || min(D_j) < min(D_all)
            if j > 1
                D_all   = cat(2, D_all, D_j);
                x_prior = x_est*var(x)/(var(x)+var(x-x_est));
            end  
            lambda = var(y-ys)/var(x-x_prior);
            [x_est, D_j] = gradientDescentWithMomentum(x_prior, x, A, ys, lambda, T, stepsize, ...
                gamma, 0, is_all_iterations);
            if norm(x-x_est) < norm(x-x_opt)
                x_opt = x_est;
            end
            j = j + 1;
        end
        D_gd_const_x0_init_x0(r, i) = norm(x-x_opt)^2/n;         
        
        % ----- 4.3 ||Ax-y||^2 + lambda ||x-xo||^2 init via 0
        x_prior = x0; % side informaiton
        j       = 1;  % number of layers 
        D_j     = 0; % obtained distortion at eat layer
        D_all   = 0; % obtained distortion at all layers
        x_opt   = zeros(size(x)); % optimal estimation
        while j <= 2 || min(D_j) < min(D_all)
            if j > 1
                D_all   = cat(2, D_all, D_j);
                x_prior = x_est*var(x)/(var(x)+var(x-x_est));
            end  
            lambda = var(y-ys)/var(x-x_prior);
            [x_est, D_j] = gradientDescentWithMomentum(x_prior, x, A, ys, lambda, T, stepsize, ...
                gamma, 1, is_all_iterations);
            if norm(x-x_est) < norm(x-x_opt)
                x_opt = x_est;
            end
            j = j + 1;
        end
        D_gd_const_x0_init_0(r, i) = norm(x-x_opt)^2/n;          
        
        % --- 5. single-layer gradient descent without side informaiton (lambda = 0)||Ax-y||^2 
        lambda = 0;
        x_est  = gradientDescentWithMomentum(zeros(size(x0)), x, A, ys, lambda, T, stepsize, gamma, ...
                    1, is_all_iterations);
        D_gd_no_priors(r, i) = norm(x-x_est)^2/n;          

           
        % --- 6. AMP ----
        % ----- 6.1 ||Ax-y||^2 + lambda ||x||^2 init via x0
        lambda = var(y-ys)/var(x);
        x_est  = AMPConstrainOnX(x0, x, A, ys, lambda, T, stepsize, gamma, 0, is_all_iterations);  
        D_amp_const_x_init_x0(r, i) = norm(x-x_est)^2/n;
        
        % --- 6.2 ||Ax-y||^2 + lambda ||x-x_0||^2 init via x0
        lambda = var(y-ys)/var(x-x0);
        x_est  = AMP(x0, x, A, ys, lambda, T, stepsize, gamma, 0, is_all_iterations);
        D_amp_const_x0_init_x0(r, i) = norm(x-x_est)^2/n;            
        
        % --- 6.3 AMP  + lambda ||x-x_0||^2 init via 0
        lambda = var(y-ys)/var(x-x0);
        x_est  = AMP(x0, x, A, ys, lambda, T, stepsize, gamma, 1, is_all_iterations);
        D_amp_const_x0_init_0(r, i) = norm(x-x_est)^2/n;         
        
        % --- 7. AMP without side informaiton (lambda = 0)||Ax-y||^2
        lambda = 0;
        x_est  = AMP(zeros(size(x0)), x, A, ys, lambda, T, stepsize, gamma, 1, is_all_iterations);
        D_amp_no_priors(r, i) = norm(x-x_est)^2/n;                
        
    end
end

%% --- averaging ----------------------------------------------------------

D_upper_bound = mean(D_upper_bound, 1); 

D_map  = mean(D_map, 1); 
D_pinv = mean(D_pinv, 1); 

D_gd_const_x_init_x0  = mean(D_gd_const_x_init_x0, 1); 
D_gd_const_x0_init_x0 = mean(D_gd_const_x0_init_x0, 1); 
D_gd_const_x0_init_0  = mean(D_gd_const_x0_init_0, 1); 
D_gd_no_priors        = mean(D_gd_no_priors, 1); 

D_amp_const_x_init_x0  = mean(D_amp_const_x_init_x0, 1); 
D_amp_const_x0_init_x0 = mean(D_amp_const_x0_init_x0, 1); 
D_amp_const_x0_init_0  = mean(D_amp_const_x0_init_0, 1); 
D_amp_no_priors        = mean(D_amp_no_priors, 1); 


%% --- visualisation ------------------------------------------------------
lw = 2;
figure;

plot(linspace(R(1), max(4.5, max(R)),100), var(x)*2.^(-2*linspace(R(1),max(4.5, max(R)),100)), ...
    'r', 'LineWidth', lw); hold on;

plot(R, D_map,  '--k', 'LineWidth', lw); hold on;
plot(R, D_pinv, '--r', 'LineWidth', lw); hold on;

plot(R, D_gd_const_x_init_x0,  'b',   'LineWidth', lw); hold on;
plot(R, D_gd_const_x0_init_x0, 'g',   'LineWidth', lw); hold on;
plot(R, D_gd_const_x0_init_0,  '--g', 'LineWidth', lw); hold on;
plot(R, D_gd_no_priors,        '--b', 'LineWidth', lw); hold on;

plot(R, D_amp_const_x_init_x0,  'm',   'LineWidth', lw); hold on;
plot(R, D_amp_const_x0_init_x0, 'c',   'LineWidth', lw); hold on;
plot(R, D_amp_const_x0_init_0,  '--c', 'LineWidth', lw); hold on;
plot(R, D_amp_no_priors,        '--m', 'LineWidth', lw); hold on;

plot(R, D_upper_bound, 'k', 'LineWidth', lw); hold on;

grid on;
legend('shannon', 'MAP',  'pinv', ...
    'GD ||Ax-y||^2 + lambda ||x||^2 init via x0', ...
    'GD ||Ax-y||^2 + lambda ||x-xo||^2init via x0', ...
    'GD ||Ax-y||^2 + lambda ||x-xo||^2init via 0', ...
    'GD ||Ax-y||^2', ...
    'AMP ||Ax-y||^2 + lambda ||x||^2 init via x0', ...
    'AMP ||Ax-y||^2 + lambda ||x-xo||^2init via x0', ...
    'AMP ||Ax-y||^2 + lambda ||x-xo||^2init via 0', ...
    'AMP ||Ax-y||^2', ...
    'upper bound' ...
);

xlabel('rate');
ylabel('distortion');
title(['n = ', num2str(n), ', ', ...
        'M = ', num2str(M), ', ', ...
        'sigma_z^2 = ', num2str(sigma^2)])
%%----------------------------------------------------------------------------
