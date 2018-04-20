function n = OU_euler_seed(N, Dt, tau, sigma,rseed)

%nu = randn(N+1,1);
n  = zeros(N,1);
randn('state',rseed);
for t=1:N-1
  %  r1=randn;
    r2=randn;
  %  k1 = n(t) - n(t)(Dt/2)*/tau;%+ sigma*(sqrt(2/tau))*r1;
    k2 = - (Dt/tau)*n(t) + sigma*(sqrt(2*Dt/tau))*r2;
    n(t+1) = n(t) + k2;
end