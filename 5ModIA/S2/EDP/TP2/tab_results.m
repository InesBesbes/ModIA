close all;
clear all;

load bcsstk27.mat;

n = size(A,1);

b = [1:n]';

fprintf('Dimension de A %d', size(A));

%%%%%%%%%%%%%%%%%% Résolution directe %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% tic;
x_ref = A \ b;
% time_direct = toc;
% 
% % Estimation du fill-in en simulant une factorisation symbolique
% [count, h, parent, post, R] = symbfact(A); 
% ALU = R + R'; % Estimation des facteurs L et U
% fillin_direct = nnz(ALU) - nnz(A);
% nnz_direct = nnz(R);
% flops_direct = 4 * nnz_direct;
% be_direct = norm(A*x_ref - b) / norm(b);
% fe_direct = 0; % Pas d'erreur car c'est la solution de référence
% mem_direct = whos('R');


%%%%%%%%%%%%%%%%%% Cholesky sans permutation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic;
L = chol(A, 'lower');
y = L \ b;
x_noperm = L' \ y;
time_noperm = toc;

nnzL_noperm = nnz(L);
fillin_noperm = nnz(L)-nnz(A)
flops_noperm =  4 * nnzL_noperm;
be_noperm = norm(A*x_noperm - b) / norm(b);
fe_noperm = norm(x_noperm - x_ref) / norm(x_ref);
mem_noperm = whos('L');


%%%%%%%%%%%%%%%%%% Cholesky avec permutation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Permutations à tester
permutations = {amd(A), symamd(A), symrcm(A), colamd(A)};
perm_names = {'Approximate Minimum Degree', 'Symmetric Approximate Minimum Degree', 'Symmetric Reverse Cuthill-McKee', 'Column Approximate Minimum Degree'};

results = struct();
results(1).name = 'Sans perm.';
results(1).time = time_noperm;
results(1).flops = flops_noperm;
results(1).fillin = fillin_noperm;
results(1).nnzL = nnzL_noperm;
results(1).memory = mem_noperm.bytes / 1024;
results(1).be = be_noperm;
results(1).fe = fe_noperm;

for i = 1:length(permutations)
    P = permutations{i};  
    B = A(P,P);
    
    % Factorisation de Cholesky avec permutation
    tic;
    L = chol(B, 'lower');
    y = L \ b(P);
    z = L' \ y;
    x = zeros(n,1);
    x(P) = z; 
    time = toc;

    nnzL = nnz(L);
    flops =  4 * nnzL;
    mem = whos('L');
    be = norm(A*x - b) / norm(b);
    fe = norm(x - x_ref) / norm(x_ref);
    
    % Stockage des résultats
    results(i+1).name = perm_names{i};
    results(i+1).time = time;
    results(i+1).flops = flops;
    results(i+1).fillin = nnz(L) - nnz(A);
    results(i+1).nnzL = nnzL;
    results(i+1).memory = mem.bytes / 1024;
    results(i+1).be = be;
    results(i+1).fe = fe;
end

% Calcul du gain en FLOPs par rapport à la résolution sans permutation
for i = 1:length(results)
    results(i).gain = (results(1).flops - results(i).flops) / results(1).flops * 100;
end

% Affichage des résultats
fprintf('\nRésultats de la factorisation :\n');
fprintf('-------------------------------------------------------------------------------------------------------------------\n');
fprintf('| %-18s | %-15s  | %-15s  | %-15s  | %-15s  | %-15s  |\n', ...
    'Paramètre', results(1).name, results(2).name, results(3).name, results(4).name, results(5).name);
fprintf('-------------------------------------------------------------------------------------------------------------------\n');
fprintf('| %-18s | %-15.4f  | %-15.4f  | %-15.4f  | %-15.4f  | %-15.4f  |\n', ...
    'Temps (s)', results(1).time, results(2).time, results(3).time, results(4).time, results(5).time);
fprintf('| %-18s | %-15.2e  | %-15.2e  | %-15.2e  | %-15.2e  | %-15.2e  |\n', ...
    'FLOPs', results(1).flops, results(2).flops, results(3).flops, results(4).flops, results(5).flops);
fprintf('| %-18s | %-15d  | %-15d  | %-15d  | %-15d  | %-15d  |\n', ...
    'Fill-in', results(1).fillin, results(2).fillin, results(3).fillin, results(4).fillin, results(5).fillin);
fprintf('| %-18s | %-15d  | %-15d  | %-15d  | %-15d  | %-15d  |\n', ...
    'NNZ de L', results(1).nnzL, results(2).nnzL, results(3).nnzL, results(4).nnzL, results(5).nnzL);
fprintf('| %-18s | %-15d  | %-15.2f  | %-15.2f  | %-15.2f  | %-15.2f  |\n', ...
    'Mémoire (KB)', results(1).memory, results(2).memory, results(3).memory, results(4).memory, results(5).memory);
fprintf('| %-18s | %-15.2e  | %-15.2e  | %-15.2e  | %-15.2e  | %-15.2e  |\n', ...
    'Backward error', results(1).be, results(2).be, results(3).be, results(4).be, results(5).be);
fprintf('| %-18s | %-15d | %-15.2e  | %-15.2e  | %-15.2e  | %-15.2e  |\n', ...
    'Forward error', results(1).fe, results(2).fe, results(3).fe, results(4).fe, results(5).fe);
fprintf('-------------------------------------------------------------------------------------------------------------------\n');
fprintf('Gain en FLOPs (comparé à la résolution sans permutation):\n');
fprintf(' Sans perm. : %.2f%%\n', results(1).gain);
fprintf(' amd  : %.2f%%\n', results(2).gain);
fprintf(' symamd     : %.2f%%\n', results(3).gain);
fprintf(' symrcm     : %.2f%%\n', results(4).gain);
fprintf(' colamd    : %.2f%%\n', results(5).gain);

