using SparseInverseSubset
using SparseArrays
using LinearAlgebra
using Test

@testset "SparseInverseSubset.jl" begin
    A = sparse([1, 5, 2, 3, 4, 2, 3, 4, 2, 3, 4, 5, 1, 4, 5],
    [1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5],
    [1.371743277584265, 0.15319636423142857, 1.2974555319793244, 0.334648233917893,
     0.5630908120922691, 0.334648233917893, 2.1530446960842218, 0.36210585506955384,
     0.5630908120922691, 0.36210585506955384, 1.5043721186197785, 0.5858861097478401,
     0.15319636423142857, 0.5858861097478401, 1.543614692728164])
    @test isposdef(A)

    n = size(A, 1)
    F = cholesky(A)
    L = sparse(F.L)
    P = F.p
    d = Vector(diag(L))
    L = tril(L * Diagonal(1 ./ d), -1)
    U = sparse(L')
    d = d.^2
    D = Diagonal(d)
    C1 = (L + D + U) .!= 0


    Z1, P1 = sparseinv(A)
    Zdense1 = inv(Matrix(A)[P1, P1])
    @test all(Z1 .≈ Zdense1 .* C1)

end
