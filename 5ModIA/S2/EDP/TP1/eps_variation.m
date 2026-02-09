close all;
clear all;

%load mat1;
load pde225_5e-1;
%load hydcar20.mat;

n = size(A,1);
fprintf('Dimension de A : %4d \n' , n);

b = (1:n)';
x0 = zeros(n, 1);
eps_values = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]; % Valeurs de eps à tester
kmax = n;

% Matrices pour stocker les résultats
iterations = zeros(length(eps_values), 2);

for i = 1:length(eps_values)
    eps = eps_values(i);
    fprintf('===== eps = %.1e =====\n', eps);

    % FOM
    fprintf('FOM\n');
    [x, flag, relres, iter, resvec] = krylov(A, b, x0, eps, kmax, 0);
    iterations(i, 1) = iter;
    fprintf('Nb iterations : %4d \n', iter);
    if(flag ~= 0)
        fprintf('Pas de convergence\n');
    end

    % GMRES
    fprintf('GMRES\n');
    [x, flag, relres, iter, resvec] = krylov(A, b, x0, eps, kmax, 1);
    iterations(i, 2) = iter;
    fprintf('Nb iterations : %4d \n', iter);
    if(flag ~= 0)
        fprintf('Pas de convergence\n');
    end
    
    pause(1);
end

% Affichage des résultats sous forme de tableau
fprintf('\nRésultats des itérations par méthode et eps :\n');
fprintf('--------------------------------------------\n');
fprintf(' eps       | FOM  | GMRES \n');
fprintf('--------------------------------------------\n');
for i = 1:length(eps_values)
    fprintf(' %.1e  | %4d | %4d \n', eps_values(i), iterations(i,1), iterations(i,2));
end
fprintf('--------------------------------------------\n');

% Affichage du graphique
figure;
loglog(eps_values, iterations(:,1), 'bo', 'LineWidth', 1, 'MarkerSize', 8); % FOM (bleu, cercles)
hold on;
loglog(eps_values, iterations(:,2), 'rs', 'LineWidth', 1, 'MarkerSize', 8); % GMRES (rouge, carrés)
hold off;

grid on;
xlabel('Tolérance \epsilon (log)', 'FontSize', 12);
ylabel('Nombre d''itérations', 'FontSize', 12);
title('Comparaison du nombre d''itérations en fonction de \epsilon (hydcar20.mat)', 'FontSize', 14);
legend('FOM', 'GMRES', 'Location', 'Best');

