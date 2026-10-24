function val_vec = Compute_GSIPM_EXP2_vec_V5B(XX, WW, GG, lG)
%-----------------------------------------
% For N-function phi(t) = exp(t^2) - 1

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

        func = @(k) obj_func_gsi_Exp2(k, wii', xii', aii', bii');
        k_start = fmincon(func, k0,[],[],[],[],constEPS,inf, [], options);
        val_vec(ii) = obj_func_gsi_Exp2(k_start, wii', xii', aii', bii');
    end
end

end

%-----------------------------------------------
% phi(t) = exp(t^2) - 1
% compute objective function & gradient function
function [f, g, h] = obj_func_gsi_Exp2(k, beta, x, a, b)

    % obj
    % --- A2
    % beta = w;
    % a = beta/lGG;
    % b = 1 + (lg/lGG);

    x2 = x.^2;
    alpha2 = k^2 * x2;
    
    b2 = b.^2;
    s12 = (a + b).^2;
    
    z12 = alpha2 ./ s12;
    z22 = alpha2 ./ b2;

    e12  = expm1(z12); % exp(z12) - 1
    e22  = expm1(z22); % exp(z22) - 1

    % Ei1 = -real(expint(-z12));
    % Ei2 = -real(expint(-z22));
    % deltaEi = Ei1 - Ei2;
    % deltaEi = -real(expint(-z12)) + real(expint(-z22));

    tmpEi = Ei_fast_smallpos([z12 z22]);
    deltaEi = tmpEi(:, 1) - tmpEi(:, 2);

    beta_over_a = beta ./ a;

    A2 = (beta_over_a/2) .* (-alpha2 .* deltaEi + s12 .* e12 - b2 .* e22);
    f = (1 + sum(A2))/k;

    % grad
    % grad = [sum(gradA2)/k] - [f/k]
    if nargout > 1
    
        x2deltaEi = x2 .* deltaEi;

        gradA2 = (-k) * beta_over_a .* x2deltaEi;
        g = (sum(gradA2) - f)/k;

        % hess
        % hessA2 = (sum(hessA2) - 2g)/k
        if nargout > 2

            hessA2 = - beta_over_a .* (x2deltaEi + (2 * x2).*(e12 - e22));
            h = (sum(hessA2) - (2*g))/k;
        end
    end
end







