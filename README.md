# SparseInverseSubset.jl

[![Build Status](https://github.com/ElOceanografo/SparseInverseSubset.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ElOceanografo/SparseInverseSubset.jl/actions/workflows/CI.yml?query=branch%3Amain)

This is a lightweight, experimental package implementing Takahashi's algorithm for 
calculating a subset of the inverse of a sparse symmetrical matrix $A$ based on its
Cholesky factorization $A = L L^T$. The partial inverse matrix $Z$ has the same sparsity 
pattern as $L + L'$ is equal to the full inverse $A^{-1}$ on these elements.

The package exports a single function, `sparseinv`, which takes a `SparseMatrixCSC` and 
returns the sparse inverse matrix and the permutation vector from the Cholesky 
decomposition:

```julia
using SparseInverseSubset
using SparseArrays
using LinearAlgebra

A = sparse(
    [1, 5, 2, 3, 4, 2, 3, 4, 2, 3, 4, 5, 1, 4, 5],
    [1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5],
    [1.371743277584265, 0.15319636423142857, 1.2974555319793244, 0.334648233917893,
        0.5630908120922691, 0.334648233917893, 2.1530446960842218, 0.36210585506955384,
        0.5630908120922691, 0.36210585506955384, 1.5043721186197785, 0.5858861097478401,
        0.15319636423142857, 0.5858861097478401, 1.543614692728164]
)
Z, P = sparseinv(A)

# check against full inverse
sparsity_pattern = Z .!= 0
Zdense = inv(Matrix(A)[P, P])
all(Z .≈ (Zdense .* sparsity_pattern)) # true
```

If the keyword argument `depermute` is set to `true`, the sparse inverse matrix is 
de-permuted before returning, so its rows and columns match those of `inv(Matrix(A))`:

```julia
Z, P = sparseinv(A; depermute=true)
sparsity_pattern = Z .!= 0
all(Z .≈ (inv(Matrix(A)) .* sparsity_pattern)) # true
```

This implementation is based on the formulas in Erisman and Tinney (1975). It has not been
optimized or tested very thoroughly, but is already be significantly faster than
calculating a dense inverse once the matrix is bigger than ~100 x 100.

## References

Erisman, A.M. and Tinney, W.F. (1975). "On computing certain elements of the inverse of a 
sparse matrix," *Communications of the ACM* 18(3) 177-179
(https://dl.acm.org/doi/10.1145/360680.360704).

Takahashi, K., Fagan, J., and Chin, M-S. (1973). "Formation of a sparse bus impedance 
matrix and its application to short circuit study. *8th PICA Conference Proceedings*, 4-6
June 1973, Minneapolis, MN.

