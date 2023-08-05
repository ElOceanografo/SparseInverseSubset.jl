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
    @inbounds for i in ii
        Z[i,j] = -sum(@inbounds U[i,k] * Z[k,j] for k in i+1:n)
        if i == j
            Z[i,j] += 1/D[i,i]
        end
    end
end

function fill_transposed_col!(Z, j)
    ii = Z.rowval[nzrange(Z, j)]
    @inbounds for i in ii
        Z[j,i] = Z[i,j]
    end
end

function get_ldup(F)
    L = sparse(F.L)
    P = F.p
    d = Vector(diag(L))
    L = tril(L * Diagonal(1 ./ d), -1)
    U = sparse(L')
    d = d.^2
    D = Diagonal(d)
    return (; L, D, U, P)
end

function get_subset(L)
    pattern = sparse(L + I + L') .!= 0
    return pattern
end

function sparseinv(A::SparseMatrixCSC)
    issymmetric(A) || error("matrix must be square and symmetrical.")
    n = size(A, 1)
    F = cholesky(A)
    L, D, U, P = get_ldup(F)
    sparsity = get_subset(L)
    Z = sparsity .* 999.0
    ii, jj, zz = findnz(Z)
    Z[n,n] = 1 / D[n,n]
    for j in sort(unique(jj), rev=true)
        fill_col!(Z, j, U, D)
        fill_transposed_col!(Z, j)
    end
    return (; Z, P)
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
