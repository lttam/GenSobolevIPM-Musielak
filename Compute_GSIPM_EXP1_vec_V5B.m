function val_vec = Compute_GSIPM_EXP1_vec_V5B(XX, WW, GG, lG)

%-----------------------------------------
% For N-function: phi(t) = exp(t) - t - 1

% XX: k x dim   : |\bar{h}(e)|
% WW: k x dim   : w_e
% GG: k x dim   : \lambda(\gamma_e)
% lG: scalar    : \lambda(\G)
% (dim: #edges)

% val_vec: k x 1
%-----------------------------------------

constEPS = 1e-10;

kk = size(XX, 1);
val_vec = zeros(kk, 1);

AA = WW/lG;
BB = 1 + (GG/lG);

options = optimoptions('fmincon', 'Algorithm','trust-region-reflective',...
    'FunctionTolerance', 1e-3, ...
    'ConstraintTolerance', 1e-3, ...
    'MaxFunctionEvaluations', 100, ...
    'MaxIterations', 100, ...
    'OptimalityTolerance', 1e-3, ...
    'StepTolerance', 1e-3, ...
    'UseParallel', true, ...
    'SpecifyObjectiveGradient',true, ...
    'HessianFcn','objective', ...
    'Display', 'off');

% k0 = 1;

for ii = 1:kk

    idNZ = abs(XX(ii, :)) > constEPS;

    if sum(idNZ) == 0
        val_vec(ii) = 0;
    else
        xii = XX(ii, idNZ);
        wii = WW(ii, idNZ);
        aii = AA(ii, idNZ);
        bii = BB(ii, idNZ);

        % solve 1d opt
        k0 = 1.0/max(xii); % stable initialization for N-function
        
        func = @(k) obj_func_gsi_Exp1(k, wii', xii', aii', bii');
        k_start = fmincon(func, k0,[],[],[],[],constEPS,inf, [], options);
        val_vec(ii) = obj_func_gsi_Exp1(k_start, wii', xii', aii', bii');
    end
end

end

%-----------------------------------------------
% phi(t) = exp(t) - t - 1
% compute objective function & gradient function
function [f, g, h] = obj_func_gsi_Exp1(k, beta, x, a, b)

    % obj
    % --- A1
    % beta = w;
    % a = beta/lGG;
    % b = 1 + (lg/lGG);

    % A1 = (beta/(2a)) * [ -alpha^2 (Ei(alpha/(a+b)) - Ei(alpha/b)) ...
    %                     + (a+b)*(alpha+a+b)*exp(alpha/(a+b)) ...
    %                     - b*(alpha+b)*exp(alpha/b) ...
    %                     - a*(2*alpha + 2*b + a) ]

    alpha = k * x;

    s1  = a + b;
    z1  = alpha ./ s1;
    z2  = alpha ./ b;

    e1  = exp(z1);
    e2  = exp(z2);

    % Ei1 = -real(expint(-z1));
    % Ei2 = -real(expint(-z2));
    % deltaEi = Ei1 - Ei2;
    % deltaEi = -real(expint(-z1)) + real(expint(-z2));

    tmpEi = Ei_fast_smallpos([z1 z2]);
    deltaEi = tmpEi(:, 1) - tmpEi(:, 2);

    s1e1 = s1 .* e1;
    be2 = b .* e2;

    term_special_A1 = - alpha.^2 .* deltaEi + s1e1 .* (alpha + s1) - be2  .* (alpha + b);

    beta_over_a = beta ./ a;
    A1 = (beta_over_a/2) .* term_special_A1 - (beta/2) .* (2*alpha + 2*b + a);

    f = (1 + sum(A1))/k;

    % grad
    % grad = [sum(gradA1)/k] - [f/k]
    if nargout > 1
            
        term_special_gradA1 = -alpha .* deltaEi + s1e1 - be2 - a;
        gradA1 = beta_over_a .* x .* term_special_gradA1;
     
        g = (sum(gradA1) - f)/k;

        % hess
        % hessA1 = (sum(hessA1) - 2g)/k
        if nargout > 2

            hessA1 = - beta_over_a .* (x.^2) .* deltaEi;
            h = (sum(hessA1) - (2*g))/k;
        end
    end
end





