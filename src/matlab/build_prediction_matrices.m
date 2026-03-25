function [Sx, Su, Sd] = build_prediction_matrices(A, B, E, Np, Nc)

nz = size(A,1);
nu = size(B,2);
nd = size(E,2);

Sx = zeros(nz*Np, nz);
Su = zeros(nz*Np, nu*Nc);
Sd = zeros(nz*Np, nd*Np);

for i = 1:Np
    row_idx = (i-1)*nz + (1:nz);

    Sx(row_idx,:) = A^i;

    for j = 1:min(i,Nc)
        col_u = (j-1)*nu + (1:nu);
        Su(row_idx, col_u) = A^(i-j) * B;
    end

    for j = 1:i
        col_d = (j-1)*nd + (1:nd);
        Sd(row_idx, col_d) = A^(i-j) * E;
    end
end
end