
function [x, flag, relres, iter, resvec] = krylov(A, b, x0, tol, maxit, type)
% Résolution de Ax = b par une méthode de Krylov

% x      : solution
% flag   : convergence (0 = convergence, 1 = pas de convergence en maxit)
% relres : résidu relatif (backward error normwise)
% iter   : nombre d'itérations
% resvec : vecteur contenant les iter normes du résidu


% A     : matrice du système
% b     : second membre
% x0    : solution initiale
% tol   : seuil de convergence (pour l'erreur inverse)
% maxit : nombre d'itérations maximum
% type  : méthode de Krylov
%         type == 0  FOM
%         type == 1  GMRES

% taille de la matrice A
n = size(A, 2);

% résidu initial
r0 = b - A*x0;

% le beta qui apparaît dans de de nombreuses formules est la norme du résidu initial
beta = norm(r0);

% initialisation du vecteur des résidus
% matlab va agrandir de lui-même le vecteur resvec (ce sera aussi le cas pour les matrices V et H)
resvec(1) = beta;

% norme de b (inutile de la calculer à chaque itération)
normb = norm(b);

% résidu relatif backward erreur normwise
relres = resvec(1) / normb;

% calcul du premier vecteur de la base de Krylov v1
V(:,1) = r0 / beta;

% taille de l'espace de Krylov == nombre d'itérations
j = 1;

% x initial (juste utile si on appelle la fonction sans avoir compléter le code : on a tous les résultats déclarés)
x = x0;

while ((relres > tol) && (j <= maxit)) % critère d'arrêt
    
    % w = Av_j
    w = A*V(:,j);
    
    % orthogonalisation (Modified Gram-Schmidt)
    for i = 1:j
        Hbarre(i,j) = V(:,i)'*w;
        w = w - Hbarre(i,j)*V(:,i);
    end
    
    % calcul de H(j+1, j) et normalisation de V(:, j+1)
    Hbarre(j+1,j) = norm(w);
    V(:, j+1) = w/Hbarre(j+1,j);
    
    % suivant la méthode
    if(type == 0)
        % FOM
        % résolution du système linéaire H.y = beta.e1
        % construction de beta.e1 (taille j)
        beta_e1 = zeros(j,1);
        beta_e1(1) = beta;
        % résolution de H.y = beta.e1 avec '\'
        H = Hbarre(1:j, 1:j);
        yj = H \ beta_e1;
        estimateur = Hbarre(j+1,j)*abs(yj(j));
    else
        % GMRES
        % résolution du problème aux moindres carrés argmin ||beta.e1 - H_barre.y||
        % construction de beta.e1 (taille j+1)
        beta_e1 = beta*speye(j+1,1);

        % résolution de argmin ||beta.e1 - H_barre.y|| avec '\'
        %yj = Hbarre\beta_e1; % Hbarre\beta_e1
        
        % Factorisation QR
        [Q, R] = qr(Hbarre, 0);

        % Transformation de beta*e1
        beta_e1_tilde = Q' * beta_e1;

        % Estimation de la norme du résidu
        estimateur = abs(beta_e1_tilde(end));

        % Résolution du système triangulaire supérieur pour y
        yj = R(1:j,1:j) \ beta_e1_tilde(1:j);
    end
    
    % calcul de l'itérée courante x à partir de x0, de V et de y
    x = x0 + V(:, 1:j)*yj;

    % calcul de la norme du résidu et rangement dans resvec
    resvec(j+1) = norm(b - A*x);

    % erreur relative de l'estimateur pour eviter de faire le calcul de
    % resvec
    %fprintf('difference : %e\n', norm(estimateur-resvec(j+1))/norm(resvec(j+1)));
    
    % calcul de la norme relative du résidu (backward error) relres
    relres = resvec(j+1)/normb;
    j= j+1;
    
end

% le nombre d'itération est j - 1 (incrément de j en fin de boucle)
iter = j-1;

% positionnement du flac suivant comment on est sortie de la boucle
if(relres > tol)
    flag = 1;
else
    flag = 0;
end
