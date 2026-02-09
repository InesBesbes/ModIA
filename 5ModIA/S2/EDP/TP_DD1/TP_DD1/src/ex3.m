clc; clear;
clear all;
close all;

%% Define the Global problem and subdomain problems
N=40;
global_domain_start = 0;
global_domain_end   = 1;

problem_global = Problem(N, N, global_domain_start, global_domain_end);
u_global = problem_global.solve_direct();


% Spatial decomposition: 
% First sub. takes Dirichlet data at x=a+d.
% Second sub. takes Dirichlet data at x=a
a=20; d=3;


%% \Omega_1
% Number of interior points 
Nx1=a+d-1;
Ny1=N;

% Location of start and the end of the subdomain
subdomain1_start    = global_domain_start; 
subdomain1_end      = (a+d)*problem_global.h;


%% \Omega_2
% Number of interior points 
Nx2=N-a;
Ny2=N;

% Location of start and the end of the subdomain
subdomain2_start    = (a)*problem_global.h;
subdomain2_end      = global_domain_end;


% % assemble subdomain problems
problem_subdomain1 = Problem(Nx1, Ny1, global_domain_start,  subdomain1_end);
problem_subdomain2 = Problem(Nx2, Ny2, subdomain2_start, subdomain2_end);



% Solver parameters
params.maxiter=500;     % maximum number of iterations
params.atol=1e-7;       % absolute tolerance on norm of residual
params.plt=0;           % plt=1 to show the subdomain solutions.


% Assign zero initial guess, and boundary value for subdomain_1
u1=[problem_global.gl zeros(N,a+d)];


% Assign zero initial guess, and boundary value for subdomain_2
u2=[zeros(N,N-a+1) problem_global.gr];


fprintf(" \n \n  Alternating Schwarz \n")
[error_alternating, resid_alternating, iters_alternating, u_global_alternating] =  alternating_Schwarz(params, problem_global, u_global, problem_subdomain1, problem_subdomain2, u1, u2, d);



fprintf(" \n \n  Parallel Schwarz \n")
[error_parallel, resid_parallel, iters_parallel, u_global_parallel] =  parallel_Schwarz(params, problem_global, u_global, problem_subdomain1, problem_subdomain2, u1, u2, d);


figure;
semilogy(resid_alternating);    
hold on;
semilogy(resid_parallel);     
grid on
xlabel('Iterations');
ylabel('||r||');
title('Alternating Schwarz vs. Parallel Schwarz')
legend('Alternating Schwarz','Parallel Schwarz');


