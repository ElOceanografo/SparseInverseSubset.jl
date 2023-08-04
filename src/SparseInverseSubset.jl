module SparseInverseSubset

using SparseArrays
using LinearAlgebra

export sparseinv


"""
Fill column n in the sparse matrix Z using Takahashi's method.
Assumes that the nth element in the column has already been calculated
"""
function fill_col!(Z, j, U, D)
    n = size(Z, 1)
    ii = Z.rowval[nzrange(Z, j)]
    ii = ii[findall(i -> (i <= j) && (i < n), ii)]
    ii = reverse(ii)
    for i in ii
        Z[i,j] = -sum(U[i,k] * Z[k,j] for k in i+1:n)
        if i == j
            Z[i,j] += 1/D[i,i]
        end
    end
end

function fill_transposed_col!(Z, j)
    Z[j, :] = Z[:, j]
end

function sparseinv(A)
    # (; L, D, P) = ldl(A)
    n = size(A, 1)
    C = cholesky(A)
    L = sparse(C.L)
    P = C.p
    d = Vector(diag(L))
    L = tril(L * Diagonal(1 ./ d), -1)
    U = sparse(L')
    d = d.^2
    D = Diagonal(d)

    C = (L + D + U) .!= 0
    Z = C .* 999.0
    n = size(A, 1)

    Z[n,n] = 1 / D[n,n]
    for j in n:-1:1
        fill_col!(Z, j, U, D)
        fill_transposed_col!(Z, j)
    end
    return Z
end


end
