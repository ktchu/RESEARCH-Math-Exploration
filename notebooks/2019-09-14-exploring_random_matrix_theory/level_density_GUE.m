% Experiment : Eigenvalues of GUE matrices
% Plot: Histogram of the eigenvalues
% Theory: Semicircle and finite "semicircle"

% TODO: calculation of exact density function is incorrect

%% Parameters
n = 3;  % matrix size
num_trials = 20000;
eigenvalue_samples = [];
bin_size = .1;

%% Experiment
for i=1:num_trials
    A = randn(n) + sqrt(-1) * randn(n);  % random complex n x n matrix
    S = (A+A') / (2 * sqrt(4*n));  % normalized matrix
    eigenvalue_samples = [eigenvalue_samples; eig(S)];
end

%% Plot
[count, x] = hist(eigenvalue_samples, -1.5:bin_size:1.5);
cla reset
bar(x, count * pi / (2*num_trials*n*bin_size), 'y');
hold on;
axis('square')
axis([-1.5 1.5 -1 2])

%% Theory
t = -1:.01:1;
plot(t, sqrt(1-t.^2), 'LineWidth', 2)  % semicircle law

% exact density function
% TODO: this calculation is broken
s = 2 * x;
phi_0 = 1 / sqrt(2^0 * 1 * sqrt(pi)) * exp(-s.^2/2);
phi_1 = 1 / sqrt(2^1 * 1 * sqrt(pi)) * exp(-s.^2/2) .* (2 * s);
phi_2 = 1 / sqrt(2^2 * 2 * sqrt(pi)) * exp(-s.^2/2) .* (4 * s.^2 - 2);
density = phi_0.^2 + phi_1.^2 + phi_2.^2;
plot(x, density, 'blue', 'LineWidth', 2)
