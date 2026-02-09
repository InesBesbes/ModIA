close all;
clear all;

load mat3.mat;

% Vérifier que la matrice A est symétrique définie positive
isSymmetric = issymmetric(A);
try
    chol(A);
    isPositiveDefinite = true;
catch
    isPositiveDefinite = false;
end

isSymmetricPositiveDefinite = isSymmetric && isPositiveDefinite;

% Affichage du résultat
disp(['La matrice est symétrique et définie positive : ', num2str(isSymmetricPositiveDefinite)]);

%%%%%%%%%%%%%%%%%%%% Factorisation symbolique %%%%%%%%%%%%%%%%%%

% n = size(A,1);
% 
% b = [1:n]';
% 
% subplot(2,3,1);
% spy(A);
% title('Original matrix A');
% 
% [count, h, parent, post, R] = symbfact(A);
% ALU = R+R';
% subplot(2,3,2)
% spy(ALU);
% title('Factors of A')
% fillin_SP_symb = nnz(ALU)-nnz(A);;
% 
% % visualisation du fill
% C = spones(A);
% CLU = spones(ALU);
% FILL = CLU-C;
% subplot(2,3,3)
% spy(FILL)
% title('Fill on original A')
% 
% % Permutation 
% P = symrcm(A); %permutation Reverse Cuthill-McKee
% 
% B = A(P,P); 
% subplot(2,3,4)
% spy(B);
% title('Permuted matrix')
% 
% [count, h, parent, post, R] = symbfact(B);
% BLU = R+R';
% subplot(2,3,5)
% spy(BLU);
% fillin_AP_symb = nnz(BLU)-nnz(A)
% title('Factors of permuted A');
% 
% B = spones(B);
% BLU = spones(BLU);
% FILL = BLU-B;
% subplot(2,3,6)
% spy(FILL);
% title('Fill on permuted A');
% 
% 
% %%%%%%%%%%%%%%%%%%%% Factorisation Cholesky %%%%%%%%%%%%%%%%%%
% 
% % Solution de référence du système Ax=b
% x_ref = A \ b; 
% 
% % Factorisation sans permutation
% tic;
% L = chol(A, 'lower');
% y = L \ b;
% x1 = L' \ y;
% time1 = toc;
% 
% fillin_SP_chol = nnz(L)-nnz(A)
% nnzL1 = nnz(L);
% flops1 = 4 * nnzL1;
% be1 = norm(A*x1-b)/norm(b); %backward error
% fe1 = norm(x1 - x_ref)/norm(x_ref); %forward error
% mem1 = whos('L'); % Espace mémoire utilisé
% 
% 
% 
% % Factorisation avec permutation
% P = symrcm(A);
% B = A(P,P); %on permute les lignes et les colonnes
% 
% tic;
% L = chol(B, 'lower');
% y = L \ b(P);
% z = L' \ y;
% x2 = zeros(n,1);
% x2(P) = z; % Dépermutation
% time2 = toc;
% 
% fillin_AP_chol = nnz(L)-nnz(A)
% nnzL2 = nnz(L);
% flops2 = 4 * nnzL2;
% be2 = norm(A*x2-b)/norm(b);
% fe2 = norm(x2 - x_ref)/norm(x_ref);
% mem2 = whos('L');
% gain = (flops1 - flops2) / flops1 * 100;
% 
% 
% fprintf("Vérification des résultats:\n");
% 
% fprintf("Nombre de termes non nuls (symbolique) %d\n", nnz(R));
% fprintf("Nombre de termes non nuls (cholesky) %d\n", nnz(L));
% 
% fprintf("Fill-in pour Cholesky sans permutation: %d\n", fillin_SP_chol);
% fprintf("Fill-in pour factorisation symbolique sans permutation: %d\n", fillin_SP_symb);
% fprintf("Fill-in pour Cholesky avec permutation: %d\n", fillin_AP_chol);
% fprintf("Fill-in pour factorisation symbolique avec permutation: %d\n", fillin_AP_symb);
% % 
% 
% 
% % Affichage des résultats pour Cholesky
% fprintf('\nRésultats de la factorisation:\n');
% fprintf('--------------------------------------------------------------\n');
% fprintf('| %-18s | %-15s | %-15s |\n', 'Paramètre', 'Sans permutation', 'Avec permutation');
% fprintf('--------------------------------------------------------------\n');
% fprintf('| %-18s | %-15.4f  | %-15.4f  |\n', 'Temps (s)', time1, time2);
% fprintf('| %-18s | %-15.2e  | %-15.2e  |\n', 'FLOPs', flops1, flops2);
% fprintf('| %-18s | %-15d  | %-15d  |\n', 'Fill-in', fillin_SP_chol, fillin_AP_chol);
% fprintf('| %-18s | %-15d  | %-15d  |\n', 'NNZ de L', nnzL1, nnzL2);
% fprintf('| %-18s | %-15.2f  | %-15.2f  |\n', 'Mémoire (KB)', mem1.bytes / 1024, mem2.bytes / 1024);
% fprintf('| %-18s | %-15.2e  | %-15.2e  |\n', 'Backward error', be1, be2);
% fprintf('| %-18s | %-15.2e  | %-15.2e  |\n', 'Forward error', fe1, fe2);
% fprintf('--------------------------------------------------------------\n');
% fprintf('Gain en FLOPs grâce à la permutation: %.2f%%\n', gain);





