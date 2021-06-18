# --- User parameters

n = 100
num_samples = 10000

# --- Imports

using BenchmarkTools
using LinearAlgebra
using Statistics

# --- Functions

"""
    Compute QR factorization of `A` using the modified Gram-Schmidt algorithm.
"""
function qr_mgs(A::Matrix)
    # Initialize Q and R
    Q = copy(A)
    R = zero(A)
    
    for j in 1:size(Q, 2)
        norm = LinearAlgebra.norm(Q[:, j])
        if norm > 0
            Q[:, j] /= norm
            R[j, j] = norm
        else
            continue
        end

        column = Q[:, j]
        for j_inner in (j + 1):size(Q, 2)
            R[j, j_inner] = Q[:, j_inner] ⋅ column
            Q[:, j_inner] -= R[j, j_inner] * column
        end
    end
    
    Q, R
end;

# --- Test qr_mgs()

n = 10
A = randn(n, n)
Q, R = qr_mgs(A)

println("Relative error Q * R: ", opnorm(A - Q * R)/opnorm(A))
println("Absolute error Q * R: ", opnorm(A - Q * R))
println("opnorm(Q' * Q): ", opnorm(transpose(Q) * Q))

det_A = det(A)
det_Q = det(Q)
det_R = det(R)
println("det(A): ", det_A)
println("det(Q): ", det_Q)
println("det(R): ", det_R)
println("Relative error det(Q * R):", (det_A - det_Q*det_R) / det_A)
println("Absolute error det(Q * R):", det_A - det_Q*det_R)

# Initialize data vectors
mgs_orthogonality_errors = zeros(num_samples)
householder_orthogonality_errors = zeros(num_samples)

mgs_det_errors = zeros(num_samples)
householder_det_errors = zeros(num_samples)

# Collect data
for i in 1:num_samples
    # Generate random matrix
    A = randn(n, n)
    det_A = det(A)

    # Compute QR factorization using modified Gram-Schmidt algorithm
    Q_mgs, R_mgs = qr_mgs(A)

    mgs_orthogonality_errors[i] = opnorm(transpose(Q_mgs) * Q_mgs - LinearAlgebra.I)

    det_R = det(R_mgs)
    mgs_det_errors[i] = abs((abs(det_A) - abs(det_R)) / det_A)
    
    # Compute QR factorization using Householder triangularization
    F_householder= qr(A)
    Q_householder = F_householder.Q
    R_householder = F_householder.R
    
    householder_orthogonality_errors[i] =
        opnorm(transpose(Q_householder) * Q_householder - LinearAlgebra.I)

    det_R = det(R_householder)
    householder_det_errors[i] = abs((abs(det_A) - abs(det_R)) / det_A)
end

# --- Orthogonality Error

println("mean(mgs_orthogonality_errors): ", mean(mgs_orthogonality_errors))
println("std(mgs_orthogonality_errors): ", std(mgs_orthogonality_errors))

println("mean(householder_orthogonality_errors): ", mean(householder_orthogonality_errors))
println("std(householder_orthogonality_errors): ", std(householder_orthogonality_errors))

# --- Determinant Error

println("mean(mgs_det_errors): ", mean(mgs_det_errors))
println("std(mgs_det_errors): ", std(mgs_det_errors))

println("mean(householder_det_errors): ", mean(householder_det_errors))
println("std(householder_det_errors): ", std(householder_det_errors))

# --- Special cases that demonstrate large loss of orthogonality with MGS

A = [0.700000 0.70711; 0.70001 0.70711]
det_A = det(A)

# MGS
Q_mgs, R_mgs = qr_mgs(A)
println("MGS orthogonality error: ", opnorm(transpose(Q_mgs) * Q_mgs - LinearAlgebra.I))
println("MGS determinant error: ", abs((abs(det_A) - abs(det(R_mgs))) / det_A))

# Householder triangularization
F_householder = qr(A)
Q_householder = F_householder.Q
R_householder = F_householder.R
println("Householder triangularization orthogonality error: ",
        opnorm(transpose(Q_householder) * Q_householder - LinearAlgebra.I))
println("Householder triangularization determinant error: ",
        abs((abs(det_A) - abs(det(R_householder))) / det_A))
