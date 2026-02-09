function [error, residual, iter, u_global] =  parallel_Schwarz(params, problem_global, u_global_exact, problem_subdomain1, problem_subdomain2, u1, u2, overlap_size)

iter=0;
converged=false;

error=zeros(params.maxiter,1);
residual=zeros(params.maxiter,1);

u1n=u1;
u2n=u2;
u_global=0*u_global_exact;

% Parallel Schwarz solver
while converged==false
    
    iter = iter+1;
    
    %% Solve on the first subdomain
    % TODO:: Assign correct BC conditions
    %     problem_subdomain1.gr = ... ;
    % TODO:: Solve the first subproblem
    %     u1n = ....;
    
    
    %% Solve on the second subdomain
    % TODO::  Assign correct BC conditions
    % problem_subdomain2.gl = u1(:,end-overlap_size);
    
    % TODO:: Solve the first subproblem
    % u2n = problem_subdomain2.solve_direct();
    
    u1=u1n;
    u2=u2n;
    
    % Merge the two contributions to form global solution
    % Pay attention to the overlapping part!!!
    % u_global= ...;
    
    
    % compute error and residual between iterate and exact solution
    error(iter+1)=problem_global.compute_error(u_global_exact, u_global);
    residual(iter)=problem_global.compute_residual(u_global);
    
    % plot subdomain solutions
    if params.plt==1
        if(iter==1)
            figure(3);
        end
        problem_subdomain1.plot_solution(u1n);
        hold on
        problem_subdomain2.plot_solution(u2n);
        hold off
        xlabel('x');ylabel('y');zlabel('Schwarz iterates ');
        title(sprintf('Iteration: %d', iter));
        disp('Press Enter to continue to the next iteration ...');
        input('', 's'); % Wait for Enter key
    end
    
    fprintf('it:     %d       || r ||  %d      || e ||  %d  \n', iter, residual(iter), error(iter));
    
    if(iter > params.maxiter || residual(iter) < params.atol)
        converged=true;
    end
    
    
end
end

