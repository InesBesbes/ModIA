clc; clear;
clear all;
close all;

%% Define the model problem
global_domain_start = 0;
global_domain_end   = 1;

% Select number of discretization points in each direction
N=40;

% Assemble the problem
problem = Problem(N, N, global_domain_start, global_domain_end);
u_global = problem.solve_direct();
problem.plot_solution(u_global);



Ns = [10, 25, 50, 100, 200];

for i = 1: size(Ns, 2)
    N = Ns(i);
    problem = Problem(N, N, global_domain_start, global_domain_end);
    
    % Extract mesh size
    hs(i) = problem.h;
    
    % TODO:: Compute condition number
    condition_numbers(i) = 1.0;
    
end



figure;
bar(1./hs, condition_numbers)
xlabel('1/h');ylabel('Cond(A)');
title('Condition number');
grid on;


