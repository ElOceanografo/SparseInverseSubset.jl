module SparseInverseSubset

using SparseArrays
using LinearAlgebra
using SuiteSparse

export sparseinv


"""
Fill column j in the sparse matrix Z using Takahashi's method, based on the upper triangle
and diagonal matrix of a Cholesky decomposition `L` and `D`. Assumes that the nth (i.e., 
the bottom) element in the column has already been calculated.
"""
function fill_col!(Z, j, U, D)
    n = size(Z, 1)
    kk = reverse(nzrange(Z, j))
    @inbounds for k in kk
        i = Z.rowval[k]
        if (i > j) || (i >= n)
            continue
        end
        Z[i,j] = -sum(@inbounds U[i,k] * Z[k,j] for k in i+1:n)
        if i == j
            Z[i,j] += 1/D[i,i]
        end
    end
end

""""
Given a symmetric sparse matrix `Z`, fill the nonzero elements in row `j` with the 
corresponding values in column `j`.
"""
function fill_transposed_col!(Z, j)
    kk = reverse(nzrange(Z, j))
    @inbounds for k in kk
        i = Z.rowval[k]
        Z[j,i] = Z[i,j]
    end
end

"""
Construct sparse Julia versions of L, D, U, and P from a CHOLMOD Cholesky factorization.
"""
function get_ldup(F)
    L = sparse(F.L)
    P = F.p
    d = Vector(diag(L))
    L = tril(L * Diagonal(1 ./ d), -1)
    U = sparse(L')
    d = d.^2
    D = Diagonal(d)
    return (L=L, D=D, U=U, P=P)
end

"""
Find the sparsity pattern of the inverse subset based on the lower Cholesky factor L.
"""
function get_subset(L)
    pattern = sparse(L + I + L') .!= 0
    return pattern
end

"""
    sparseinv(A::SparseMatrixCSC[; depermute=false])
    sparseinv(F::SuiteSparse.CHOLMOD.Factor[; depermute=false])

Calculate the inverse subset of the symmetrical sparse matrix `A` using Takahashi's
method. Returns a `NamedTuple` with fields `Z` and `P`, where `Z` is the partial inverse
of `A` and `P` is a permutation vector. If the Cholesky factorization of `A` has already
been computed, that can be supplied instead.

If `L` is the lower-triangular Cholesky factor of `A`, then `Z` will have the same 
sparsity pattern as `L + L' + I`. Each nonzero entry `Z[i, j]` will be equal to the 
corresponding entry in `inv(Matrix(A)[P, P])` (up to machine precision).

If `depermute=true`, the matrix `Z` is de-permuted before returning, so that 
`Z[i, j] == inv(Matrix(A))[i,j]`.

Erisman, A.M. and Tinney, W.F. (1975). "On computing certain elements of the inverse of a 
sparse matrix," *Communications of the ACM* 18(3) 177-179
(https://dl.acm.org/doi/10.1145/360680.360704).

Takahashi, K., Fagan, J., and Chin, M-S. (1973). "Formation of a sparse bus impedance 
matrix and its application to short circuit study. *8th PICA Conference Proceedings*, 4-6
June 1973, Minneapolis, MN.
"""
function sparseinv(A::SparseMatrixCSC; depermute=false)
    issymmetric(A) || error("matrix must be square and symmetrical.")
    F = cholesky(A)
    return sparseinv(F; depermute=depermute)
end

function sparseinv(F::SuiteSparse.CHOLMOD.Factor; depermute=false)
    L, D, U, P = get_ldup(F)
    n = size(L, 1)
    sparsity = get_subset(L)
    Z = float(sparsity)
    ii, jj, zz = findnz(Z)
    Z[n,n] = 1 / D[n,n]
    for j in sort(unique(jj), rev=true)
        fill_col!(Z, j, U, D)
        fill_transposed_col!(Z, j)
    end
    if depermute
        return (Z = Z[invperm(P), invperm(P)], P = P)
    else
        return (Z=Z, P=P)
    end
end

# function sparseinv(A::SparseMatrixCSC, sparsity::SparseMatrixCSC)
#     n = size(A, 1)
#     F = cholesky(A)
#     L, D, U, P = get_ldup(F)
#     Z = sparsity .* 999.0
#     Z[n,n] = 1 / D[n,n]
#     for j in n:-1:1
#         fill_col!(Z, j, U, D)
#         fill_transposed_col!(Z, j)
#     end
#     return (; Z, P)
# end

# function sparseinv(F, sparsity)
#     L, D, U, P = get_ldup(F)
#     sparsity = get_subset(L)
#     Z = sparsity .* 999.0
#     Z[n,n] = 1 / D[n,n]
#     for j in n:-1:1
#         fill_col!(Z, j, U, D)
#         fill_transposed_col!(Z, j)
#     end
#     return (; Z, P)
# end


end
