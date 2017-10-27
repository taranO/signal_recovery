%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Convergence rate of recovering the original signal x from a quantized
% sparse representation y=Q(phi(A)) with a given noisy observation 
% x_0 = x + z
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
M  = 500; % number of samples
s  = ceil(0.09*M); % sparsity

m  = 10000;

% gradient descent params
T = 500;          % number of iterations
is_all_iterations = 1; % 1 - finish after T iteration; 0 - finish after convergence;
stepsize = 1;     % step size
gamma    = 0.005; % backtracking line search parameter
is_zero  = 0;     % start initialisation type: 0 - start from prior; 1 - strart from zero 



Run     = 3; % number of runs

sigma   = 1; % noise standard deviation

% --- Functions -----------------------------------------------------------
thresholds  = @(sortedData, K) sortedData(K, :);
kLagestMask = @(P, K) bsxfun(@ge,P, thresholds(sort(P, 'descend'), K));

%% --- Process ------------------------------------------------------------
X_dbase = randn(m,n); X_dbase = X_dbase - repmat(mean(X_dbase),m,1);
X  = X_dbase;
%--------------------------------------------------------------------------
% --- estimation of normalized coefficient for pseudo-inverse solution ----
A    = randn(M, n); % transform matrix
if M < n
    A = orth(A')';
else
    A = orth(A);
end

fX = X*A';
for ii=1:m
    fX(ii, :) = sign(fX(ii, :)').*kLagestMask(abs(fX(ii, :)'), s); 
end

[U, S, V] = svd(A);
X_hat     = fX*(V*pinv(S)*U')';
b_ps      = trace(X'*X_hat)/trace(X_hat'*X_hat);
%--------------------------------------------------------------------------

for r=1:Run
    A    = randn(M, n); % transform matrix
    if M < n
        A    = orth(A')';
    else
        A    = orth(A);
    end

    z  = randn(n, 1)*sigma; % noise
    x  = randn(n, 1);       % original signal
    x0 = x + z;             % side informaiton
    y  = A*x;   
    ys = sign(y).*kLagestMask(abs(y), s); % quantized-sparsified representation of x

    sx = var(x);  % variance of original signal
    sz = sigma^2; % variance of noise

    % --- 1. MAP estimation ----
    x_est       = x0*sx/(sx+sz);
    D_map(r, :) = repmat(norm(x-x_est)^2/n, 1, T-1);

    % --- 2. pseudo-inverse ----
    x_est        = b_ps*(pinv(A'*A)*A'*ys);
    D_pinv(r, :) = repmat(norm(x-x_est)^2/n, 1, T-1);

    % --- 4. multi-layer gradient descent ----
    
    % ----- 4.1 ||Ax-y||^2 + lambda ||x||^2 init via x0
    x_prior = x0; % side informaiton
    j       = 1;  % number of layers 
    D_j     = 0;  % obtained distortion at eat layer
    D_all   = []; % obtained distortion at all layers
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
    D_gd_const_x_init_x0(r, :) = (D_all.^2)./n;    
    
    % ----- 4.2. ||Ax-y||^2 + lambda ||x-x0||^2 init via x0
    x_prior = x0; % side informaiton
    j       = 1;  % number of layers 
    D_j     = 0;  % obtained distortion at eat layer
    D_all   = []; % obtained distortion at all layers
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
    D_gd_const_x0_init_x0(r, :) = (D_all.^2)./n;    

    % ----- 4.3. ||Ax-y||^2 + lambda ||x-xo||^2 init via 0
    x_prior = x0; % side informaiton
    j       = 1;  % number of layers 
    D_j     = 0;  % obtained distortion at eat layer
    D_all   = []; % obtained distortion at all layers
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
    D_gd_const_x0_init_0(r, :) = (D_all.^2)./n;  

   
    % --- 5. AMP ----
    % ----- 5.1 ||Ax-y||^2 + lambda ||x||^2 init via x0
    lambda = var(y-ys)/var(x);
    [x_est, E]  = AMPConstrainOnX(x0, x, A, ys, lambda, T, stepsize, gamma, 0, is_all_iterations);  
    D_amp_const_x_init_x0(r, :) = (E.^2)./n;   
    
    % --- 5.2 ||Ax-y||^2 + lambda ||x-x_0||^2 init via x0
    lambda = var(y-ys)/var(x-x0);
    [x_est, E]  = AMP(x0, x, A, ys, lambda, T, stepsize, gamma, 0, is_all_iterations);
    D_amp_const_x0_init_x0(r, :) = (E.^2)./n;      

    % --- 5.3 AMP  + lambda ||x-x_0||^2 init via 0
    lambda = var(y-ys)/var(x-x0);
    [x_est, E]  = AMP(x0, x, A, ys, lambda, T, stepsize, gamma, 1, is_all_iterations);
    D_amp_const_x0_init_0(r, :) = (E.^2)./n;   
  
end
%% --- averaging ----------------------------------------------------------

D_map  = average(D_map); 
D_pinv = average(D_pinv); 

D_gd_const_x_init_x0  = average(D_gd_const_x_init_x0); 
D_gd_const_x0_init_x0 = average(D_gd_const_x0_init_x0); 
D_gd_const_x0_init_0  = average(D_gd_const_x0_init_0); 

D_amp_const_x_init_x0  = average(D_amp_const_x_init_x0); 
D_amp_const_x0_init_x0 = average(D_amp_const_x0_init_x0); 
D_amp_const_x0_init_0  = average(D_amp_const_x0_init_0); 

%% --- visualisation ------------------------------------------------------
lw = 2;
figure;

plot(D_map, 'k', 'LineWidth', lw); hold on;
plot(D_pinv, 'c', 'LineWidth', lw); hold on;

plot(D_gd_const_x_init_x0,  'b',   'LineWidth', lw); hold on;
plot(D_gd_const_x0_init_x0, 'r',   'LineWidth', lw); hold on;
plot(D_gd_const_x0_init_0,  'g', 'LineWidth', lw); hold on;

plot(D_amp_const_x_init_x0,  '--b',   'LineWidth', lw); hold on;
plot(D_amp_const_x0_init_x0, '--r',   'LineWidth', lw); hold on;
plot(D_amp_const_x0_init_0,  '--g', 'LineWidth', lw); hold on;

grid on; 
legend('MAP(xo)', 'A^{+}y', ...
    'GD: ||y-Ax||^2+lambda||x||^2 init via x0', ...
    'GD: ||y-Ax||^2+lambda||x-xo||^2 init via x0', ...
    'GD: ||y-Ax||^2+lambda||x-xo||^2 init via 0', ...
    'AMP: ||y-Ax||^2+lambda||x||^2 init via x0', ...
    'AMP: ||y-Ax||^2+lambda||x-xo||^2 init via x0', ...
    'AMP: ||y-Ax||^2+lambda||x-xo||^2 init via 0' ...
);
xlabel('iteration');
ylabel('distortion');
title(['n = ', num2str(n), ', ', ...
        'M = ', num2str(M), ', ', ...
        'sigma_z^2 = ', num2str(sigma^2)]);
































