function [betak, Rk] = FindBetaR(L1,Zmin)

a = 1; % radius of curvature of the apex of the spheroid
L = L1*a; % long semi-axis of the spheroid
zmin = Zmin*a; % corresponds to 0.6nm for a=30nm
%zmax = 6*a; % corresponds to 180nm
zmax = 10*a;
Nz = 200; % number of grid points for tip sample separation
Nz1 = ceil((Nz-1)/2); % half of the points lie between ztip=0.02-0.5a
ztip = [linspace((zmin)^(1/3),(0.5-0.5/Nz1)^(1/3),Nz1).^3,...
                                 linspace(0.5, zmax, Nz-Nz1)];
N=200; % dimension of eigenproblem
betak = zeros(N,Nz); % array storing poles
Rk = zeros(N,Nz); % array storing residues


parfor h = 1 : Nz % for loop over all ztip
[betak_, Rk_] = PolesResidues(ztip(h), a, L, N);

betak(:,h) = betak_ ;
Rk(:,h) = Rk_ ;
end
