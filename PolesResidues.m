function [betak, Rk] = PolesResidues(ztip, a, L, N)
W = sqrt(a * L); % short semi-axis
F = sqrt(L^2 - W^2); % focal length
xi0 = L / F; % inverse eccentricity
p = zeros(1, N - 1); % array storing the ratios p
p(1) = 2 / (3 * xi0 - 1 / xi0); % the n = 1 term
for n = 2: N - 1
p(n) = (n + 1) / ((2 * n + 1) * xi0 - n * p(n - 1));

end

qinf = 1/(xi0 + sqrt(xi0^2 - 1)); % asympototic limit at large n
nmax = ceil(-N / (0.01 + log(qinf))); % some large n > N
tmp = qinf;

for n = N + nmax - 1: -1: N - 1
tmp = (n + 1) / ((2 * n + 3) * xi0 - (n + 2) * tmp);
% disp(n)
end
q = zeros(1, N-1); % array storing the ratios q
q(N-1) = tmp;

for n = N - 2: -1: 1
q(n) = (n + 1) / ((2 * n + 3) * xi0 - (n + 2) * q(n + 1));

end

Q1_P1 = 0.5 * log((xi0 + 1) / (xi0 - 1)) - 1 / xi0; % Q1/P1
Lambda = 4 ./ (3: 2: 2 * N + 1).*cumprod([Q1_P1, q .* p]);


H = zeros(N, N); % array storing elements of H

for n = 1: N
H(n, n) = H_quad(n, n); % diagonal elements

end
for n = 1: N-1
H(n, n + 1) = H_quad(n, n + 1); % next to diagonal elements
end
for n = 1: N-2
H(n, N) = H_quad(n, N); % N-th column
end

for g = N-2: -1: 2
for l = 1: g-1
n = g-l+1;
n2 = N-l;
H(n-1, n2) = H(n + 1, n2) - (2 * n + 1) / (2 * n2 + 1)...
             * (H(n, n2 + 1) - H(n, n2 - 1));

end
end

irtLambda = 1 ./ sqrt(Lambda); % square root of inverse Lambda

for n = 1: N
for l = n: N
H(n, l) = H(n, l) * irtLambda(n) * irtLambda(l);

H(l, n) = H(n, l);

end
end

[V, betak] = eig(eye(N), H); % solve the eigenproblem

betak=diag(betak); % turn diagonal matrix into a column

M = (V' * H * V);

Rk=4/9*F^3/Lambda(1)*abs(V(1,:)').^2./diag(M);

[~,index] = sort(1./betak,'descend');
betak=betak(index);
Rk=Rk(index);

function H_nl = H_quad(n, l)
nn = n + 0.5;
ll = l + 0.5;

zF = xi0 + ztip / F; % this is (ztip+L)/F

%f = @(x) 2 * pi .* exp(-2 * (zF - 1) * x) ./ (x + eps) .* besseli(nn, x, 1) .* besseli(ll, x, 1);

%H_nl = quadgk(f, 0, inf);

H_nl = quadgk(@(x)intH(x, nn, ll), 0, inf);

function res = intH(x, nn, ll)
B = besseli(nn, x, 1) .* besseli(ll, x, 1);
res = 2 * pi .* exp(-2 * (zF - 1) * x) ./ (x + eps) .* B;

end
end % end of H quad
%H_QUAD = @H_quad;
end % end of PolesResidues