# SparseInverseSubset

[![Build Status](https://github.com/ElOceanografo/SparseInverseSubset.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ElOceanografo/SparseInverseSubset.jl/actions/workflows/CI.yml?query=branch%3Amain)

This is a work-in-progress lightweight package implementing Takahashi's algorithm for 
calculating a subset of the inverse of a sparse matrix based on its Cholesky factorization.

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
Z = sparseinv(A)

# check against full inverse
F = cholesky(A)
P = F.P
L = sparse(F.L)
sparsity_pattern = (L + L') .!= 0
Zdense = inv(Matrix(A)[P, P])
all(Z .≈ (Zdense .* sparsity_pattern))
```

This implementation is based on the formulas in Erisman and Tinney (1975), "On computing
certain elements of the inverse of a sparse matrix," *Communications of the ACM* 18(3) 
177-179.
