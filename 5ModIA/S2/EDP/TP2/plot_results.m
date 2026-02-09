close all;
clear all;

load mat3.mat;

n = size(A,1);

figure('Position', [100, 100, 1300, 1300]);

%%%%%%%%%%%%%%% Résultats pour la matrice originale %%%%%%%%%%%%%%%
subplot(4,3,1);
spy(A);
title('Original matrix A', 'FontSize', 8);

[count, h, parent, post, R] = symbfact(A);
ALU = R+R';
subplot(4,3,2);
spy(ALU);
title('Factors of A', 'FontSize', 8);

% Visualisation du fill-in
C = spones(A);
CLU = spones(ALU);
FILL = CLU - C;
subplot(4,3,3);
spy(FILL);
title('Fill on original A', 'FontSize', 8);

%%%%%%%%%%%%%%% Résultats factorisation symbolique avec permutation Symmetric Approximate Minimum Degree %%%%%%%%%%%%%%%%

P = symamd(A);
B = A(P,P);

subplot(4,3,4);
spy(B);
title('symamd(A) permuted matrix', 'FontSize', 8);

[count, h, parent, post, R] = symbfact(B);
BLU = R+R';
subplot(4,3,5);
spy(BLU);
title('Factors of symamd(A)', 'FontSize', 8);

% Visualisation du fill-in
B_sparse = spones(B);
BLU_sparse = spones(BLU);
FILL = BLU_sparse - B_sparse;
subplot(4,3,6);
spy(FILL);
title('Fill on symamd(A)', 'FontSize', 8);

%%%%%%%%%%%%%%% Résultats factorisation symbolique avec permutation Symmetric Reverse Cuthill-McKee %%%%%%%%%%%%%%%%
P = symrcm(A);
B = A(P,P);

subplot(4,3,7);
spy(B);
title('symrcm(A) permuted matrix', 'FontSize', 8);

[count, h, parent, post, R] = symbfact(B);
BLU = R+R';
subplot(4,3,8);
spy(BLU);
title('Factors of symrcm(A)', 'FontSize', 8);

% Visualisation du fill-in
B_sparse = spones(B);
BLU_sparse = spones(BLU);
FILL = BLU_sparse - B_sparse;
subplot(4,3,9);
spy(FILL);
title('Fill on symrcm(A)', 'FontSize', 8);

%%%%%%%%%%%%%%% Résultats factorisation symbolique avec permutation Approximate Minimum Degree %%%%%%%%%%%%%%%%

P = amd(A);
B = A(P,P);

subplot(4,3,10);
spy(B);
title('amd(A) permuted matrix', 'FontSize', 8);

[count, h, parent, post, R] = symbfact(B);
BLU = R+R';
subplot(4,3,11);
spy(BLU);
title('Factors of amd(A)', 'FontSize', 8);

% Visualisation du fill-in
B_sparse = spones(B);
BLU_sparse = spones(BLU);
FILL = BLU_sparse - B_sparse;
subplot(4,3,12);
spy(FILL);
title('Fill on amd(A)', 'FontSize', 8);



% saveas(gcf, 'fill_in_analysi_bcsstk27.png');   
% print('fill_in_analysis', '-dpng', '-r300');

% fprintf('Image enregistrée sous "fill_in_analysis.png" en haute résolution.\n');
