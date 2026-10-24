function yv = Ei_fast_smallpos(xv)
% For fast computation of special function Ei

tol = 1e-3;
xSmall = 1.0;
maxIter = 10;
constGammaE = 0.57721566490153286060651209;

yv = zeros(size(xv), 'like', xv);

small = (xv <= xSmall);
% ---- Small x: convergent series ----
if any(small)
    xs = xv(small);

    % Ei(x) = gamma + ln x + sum_{k=1}^\infty x^k/(k k!)
    s   = constGammaE + log(xs);
    term = xs;      % x^1/1!
    k    = 1;
    for it = 1:maxIter
        add = term ./ k;     % x^k/(k k!)
        s   = s + add;
        if all(abs(add) <= tol*(1+abs(s))), break; end
        k    = k + 1;
        term = term .* (xs ./ k);  % update x^k/k!
    end
    yv(small) = s;
end

large   = ~small;
% ---- Large-range: compiled builtin via identity Ei(x) = -real(expint(-x)) ----
if any(large)
    xm = xv(large);
    yv(large) = -real(expint(-xm));
end

end



