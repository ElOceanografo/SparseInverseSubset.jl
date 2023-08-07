using SparseInverseSubset
using SparseArrays
using LinearAlgebra
using Test
using Random

function random_sparse_pdmat(n, p)
    A = sprandn(n, n, p)
    return A'A+I
end


@testset verbose=true "SparseInverseSubset.jl" begin
    A1 = sparse(
        [1, 5, 2, 3, 4, 2, 3, 4, 2, 3, 4, 5, 1, 4, 5],
        [1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5],
        [1.371743277584265, 0.15319636423142857, 1.2974555319793244, 0.334648233917893,
            0.5630908120922691, 0.334648233917893, 2.1530446960842218, 0.36210585506955384,
            0.5630908120922691, 0.36210585506955384, 1.5043721186197785, 0.5858861097478401,
            0.15319636423142857, 0.5858861097478401, 1.543614692728164]
    )

    @testset "Numerics" begin
        Random.seed!(1)
        A2 = [random_sparse_pdmat(100, 0.02) for _ in 1:10]
        A3 = [random_sparse_pdmat(1000, 0.001) for _ in 1:10]

        for A in [[A1]; A2; A3]
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

            for X in [A, F]
                Z1, P1 = sparseinv(X)
                Zdense1 = inv(Matrix(A)[P1, P1])
                @test all(Z1 .≈ Zdense1 .* C1)
                Z2, P2 = sparseinv(X, depermute=true)
                Zdense2 = inv(Matrix(A))
                C2 = Z2 .!= 0
                @test all(Z2 .≈ Zdense2 .* C2)
            end
        end
    end
    
    @testset "Errors" begin
        @test_throws ErrorException sparseinv(A1[:, 1:3])
        @test_throws ErrorException sparseinv(sprand(5, 5, 0.5))
        @test_throws MethodError sparseinv(rand(5, 5))
    end
end
