# --- Imports

import Distributions
using Distributions: MvNormal, Uniform, ContinuousUnivariateDistribution, cdf, pdf
using HypothesisTests
using LinearAlgebra
using Plots
using Random

# --- Generate sample of vectors drawn from a uniform distribution on a unit circle

n = 2
num_samples_2d = 10000

# Generate vectors
dist = MvNormal(zeros(n), ones(n))
num_vectors = num_samples_2d
vectors = rand(dist, num_vectors)
for i = 1:num_vectors
    vectors[:, i] /= norm(vectors[:, i])
end

# dist = Uniform(-π, π)
# num_vectors = num_samples_2d
# theta = rand(dist, num_vectors)
# vectors = transpose(hcat(cos.(theta), sin.(theta)))

# --- Plot histogram of angles

thetas = map(i -> atan(vectors[:, i][2], vectors[:, i][1]), 1:num_vectors)

num_hist_bins = 25
hist_bins = range(-pi, pi; length=num_hist_bins)
hist = histogram(thetas; bins=hist_bins, normalize=true)
plt = plot(hist)

expected_pdf_2d = 0.5 / π
println("Expected density: $(expected_pdf_2d)")

# Display plot
display(plt)

# --- Perform test for uniformity on the circle

# Analytical formula for distribution over angle from x-axis
struct ThetaDistribution2D <: ContinuousUnivariateDistribution end
Distributions.pdf(dist::ThetaDistribution2D, x::Real) = 0.5 / π
Distributions.cdf(dist::ThetaDistribution2D, x::Real) = 0.5 * (x + π) / π

dist = ThetaDistribution2D()

# Perform Anderson-Darling Test
test_results = OneSampleADTest(thetas, dist)
println(test_results)

test_results = ExactOneSampleKSTest(thetas, dist)
println(test_results)

test_results = ApproximateOneSampleKSTest(thetas, dist)
println(test_results)

# --- Generate sample of vectors drawn from a uniform distribution on a unit sphere

n = 3
num_samples_3d = 50000

# Generate vectors
dist = MvNormal(zeros(n), ones(n))
num_vectors = num_samples_3d
vectors = rand(dist, num_vectors)
for i = 1:num_vectors
    vectors[:, i] /= norm(vectors[:, i])
end

# --- Plot histogram of angles

thetas = map(i -> atan(norm(vectors[:, i][2:end]), vectors[:, i][1]), 1:num_vectors)

num_hist_bins = 50
hist_bins = range(0, π; length=num_hist_bins)
hist = histogram(thetas; bins=hist_bins, normalize=true)
plt = plot(hist)

expected_pdf_3d(x) = 0.5 * sin(x)
plt = plot!(expected_pdf_3d, 0, π)

# Display plot
display(plt)

# --- Perform test for uniformity on the sphere

# Analytical formula for distribution over angle from x-axis
struct ThetaDistribution3D <: ContinuousUnivariateDistribution end
Distributions.cdf(dist::ThetaDistribution3D, x::Real) = 0.5 * (1 - cos(x))

dist = ThetaDistribution3D()

# Perform Anderson-Darling Test
test_results = OneSampleADTest(thetas, dist)
println(test_results)

test_results = ExactOneSampleKSTest(thetas, dist)
println(test_results)

test_results = ApproximateOneSampleKSTest(thetas, dist)
println(test_results)

# --- Generate sample of vectors drawn from a uniform distribution on a unit sphere

n = 20
num_samples = 50000

# Generate vectors
dist = MvNormal(zeros(n), ones(n))
num_vectors = num_samples
vectors = rand(dist, num_vectors)
for i = 1:num_vectors
    vectors[:, i] /= norm(vectors[:, i])
end

# --- Plot histogram of angles

thetas = map(i -> atan(norm(vectors[:, i][2:end]), vectors[:, i][1]), 1:num_vectors)

num_hist_bins = 50
hist_bins = range(0, π; length=num_hist_bins)
hist = histogram(thetas; bins=hist_bins, normalize=true)
plt = plot(hist)

expected_pdf_nd(x) = (2^(n-2) / π / binomial(n-2, (n-2) ÷ 2)) * sin(x)^(n-2)
plt = plot!(expected_pdf_nd, 0, π)

# Display plot
display(plt)

# --- Perform test for uniformity on the hypersphere

# Analytical formula for distribution over angle from x-axis
struct ThetaDistributionND <: ContinuousUnivariateDistribution end

function Distributions.cdf(dist::ThetaDistributionND, x::Real)
    k = n-2
    sgn = (-1)^(k÷2)
    value = 0
    for j = 0:(k÷2 - 1)
        value += (-1)^j * binomial(k, j) * sin((k - 2*j) * x) / (k - 2*j)
    end
    value *= 2
    value += sgn * binomial(k, k÷2) * x  # contribution from i = k÷2 term
    value *= sgn / π / binomial(k, k ÷ 2)

    return value
end

dist = ThetaDistributionND()

# Perform Anderson-Darling Test
test_results = OneSampleADTest(thetas, dist)
println(test_results)

test_results = ExactOneSampleKSTest(thetas, dist)
println(test_results)

test_results = ApproximateOneSampleKSTest(thetas, dist)
println(test_results)

# --- Generate sample of vectors drawn from a uniform distribution on a unit sphere

n = 45
num_samples = 50000

# Generate vectors
dist = MvNormal(zeros(n), ones(n))
num_vectors = num_samples
vectors = rand(dist, num_vectors)
for i = 1:num_vectors
    vectors[:, i] /= norm(vectors[:, i])
end

# --- Plot histogram of angles

thetas = map(i -> atan(norm(vectors[:, i][2:end]), vectors[:, i][1]), 1:num_vectors)

num_hist_bins = 50
hist_bins = range(0, π; length=num_hist_bins)
hist = histogram(thetas; bins=hist_bins, normalize=true)
plt = plot(hist)

expected_pdf_nd(x) = (2^(n-2) / π / binomial(n-2, (n-2) ÷ 2)) * sin(x)^(n-2)
plt = plot!(expected_pdf_nd, 0, π)

# Display plot
display(plt)

# --- Perform test for uniformity on the hypersphere

# Analytical formula for distribution over angle from x-axis
struct ThetaDistributionND <: ContinuousUnivariateDistribution end

function Distributions.cdf(dist::ThetaDistributionND, x::Real)
    k = n-2
    value = 0
    normalization = 0
    for j = 0:k
        coef = (-1)^j * binomial(k, j) / (k - 2*j)
        value += coef * (1 - cos((k - 2*j) * x))
        normalization += coef
    end
    normalization = 0.5 / normalization
    value *= normalization

    return value
end

dist = ThetaDistributionND()

# Perform Anderson-Darling Test
test_results = OneSampleADTest(thetas, dist)
println(test_results)

test_results = ExactOneSampleKSTest(thetas, dist)
println(test_results)

test_results = ApproximateOneSampleKSTest(thetas, dist)
println(test_results)
