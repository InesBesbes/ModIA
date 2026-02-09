classdef Problem
    properties
        Nx         % Number of interior mesh points in x-direction
        Ny         % Number of interior mesh points in y-direction
        h          % Mesh size
        
        x_start    % Start point in x-direction
        x_end      % End point in x-direction
        
        x          % Finite difference mesh in x-direction (including boundary)
        y          % Finite difference mesh in y-direction (including boundary)
        xi          % Finite difference mesh in x-direction (excluding boundary)
        yi         % Finite difference mesh in y-direction (excluding boundary)
        
        gl         % Boundary data on the left
        gr         % Boundary data on the right
        
        A          % Stiffness matrix
        f          % Right hand side
        z          % Zero vector used for nicer ploting
    end
    
    methods
        function obj = Problem(Nx, Ny, x_start, x_end)
            % Constructor to initialize problem parameters and data
            obj.Nx = Nx;
            obj.Ny = Ny;
            
            obj.x_start     = x_start;
            obj.x_end       = x_end;
            
            % We assume the same spacing in both directions
            obj.h = 1 / (obj.Ny + 1);
            
            % Finite difference mesh, including boundary
            obj.y = (0 : 1 / (obj.Ny + 1) : 1);
            obj.x = (obj.x_start : 1 / (obj.Ny + 1) : obj.x_end);
            
            % Finite difference mesh, excluding boundary
            obj.xi=obj.x(2:end-1);
            obj.yi=obj.y(2:end-1);
            
            % Define Rhs
            obj.f = obj.defineSourceTerm();
            
            % Left and right BC
            obj.gl = zeros(obj.Ny, 1);
            obj.gr = zeros(obj.Ny, 1);
            
            % Assembly of stiffness matrix
            obj.A = obj.assembly_A(obj.h, obj.Nx, obj.Ny);
            
            obj.z=zeros(1,obj.Nx+2);
        end
        
        
        function f = defineSourceTerm(obj)
            % Define the source term for the model problem
            f = zeros(size(obj.yi, 2), size(obj.xi, 2)+2);
            f((obj.yi > 0.4 & obj.yi < 0.6), :) = 50;
            f = f(:, 2:end-1);  % Restrict to the interior of Omega
        end
        
        
        function A = assembly_A(~, h, Nx, Ny)
            % Assembly of finite difference approximation of -Delta in 2D
            Ax = (1 / h^2) * spdiags([-ones(Nx, 1), 2 * ones(Nx, 1), -ones(Nx, 1)], -1:1, Nx, Nx);
            Ay = (1 / h^2) * spdiags([-ones(Ny, 1), 2 * ones(Ny, 1), -ones(Ny, 1)], -1:1, Ny, Ny);
            A = kron(Ax, speye(Ny)) + kron(speye(Nx), Ay);
        end
        
        function u = solve_direct(obj)
            % Solve the boundary value problem represented by A and force
            % term f using direct solver
            obj.f(1:obj.Ny, 1) = obj.f(1:obj.Ny, 1) + obj.gl / obj.h^2;
            obj.f(1:obj.Ny, end) = obj.f(1:obj.Ny, end) + obj.gr / obj.h^2;
            u = obj.A \ obj.f(:);
            u = reshape(u, obj.Ny, obj.Nx);
            u = [obj.gl u obj.gr];
        end
         
        function [u, iter] = solve_cg(obj)
            % Solve the boundary value problem represented by A and force
            % term f using matlab's cg solver
            obj.f(1:obj.Ny, 1) = obj.f(1:obj.Ny, 1) + obj.gl / obj.h^2;
            obj.f(1:obj.Ny, end) = obj.f(1:obj.Ny, end) + obj.gr / obj.h^2;
            
            % initial guess
            u = 0.0*obj.f(:);
            iter=0;
            
            % Define tolerance
            tol=1e-10;
            maxit=size(obj.A,1);
            
            % Call CG solver
            % [u,~,~,iter] = ...;
            
            u = reshape(u, obj.Ny, obj.Nx);
            u = [obj.gl u obj.gr];
        end
        
        
        function [u, iter] = solve_pcg(obj)
            % Solve the boundary value problem represented by A and force
            % term f using matlab's preconditioned cg solver
            obj.f(1:obj.Ny, 1) = obj.f(1:obj.Ny, 1) + obj.gl / obj.h^2;
            obj.f(1:obj.Ny, end) = obj.f(1:obj.Ny, end) + obj.gr / obj.h^2;
            
            % Define tolerance
            tol=1e-10;
            maxit=size(obj.A,1);
            
            % TODO:: Get Cholesky preconditioner
            M = eye(size(obj.A)); 
            % M = ....;
            
            % TODO:: Call pcg solver with preconditioner M
            u = 0* obj.f(:); 
            iter = 0; 
            % [u,~,~,iter] = ..... ;
            
            u = reshape(u, obj.Ny, obj.Nx);
            u = [obj.gl u obj.gr];
        end
        
        
        % Plot solution on the mesh
        function [] = plot_solution(obj, u)
            mesh(obj.x, obj.y, [obj.z;u;obj.z]);
            xlabel('x');ylabel('y');
        end
        
        % Compute residual
        function [residual] = compute_residual(obj, u_approx)
            % We consider just inner part of the computational domain, as
            % BC are enforced exactly
            u_approx  = u_approx(:, 2:obj.Ny+1);
            residual = norm(obj.f(:) - obj.A*u_approx(:),2);
        end
        
        
        % Compute error
        function [error] = compute_error(~, u_exact, u_approx)
            error = norm(u_exact-u_approx,2);
        end
        
    end
end

