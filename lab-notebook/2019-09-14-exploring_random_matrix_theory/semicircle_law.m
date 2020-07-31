% Experiment : Gaussian Random Symmetric Eigenvalues
% Plot: Histogram of the eigenvalues
% Theory: Semicircle as n−>infinity

%% Parameters
n = 1000;  % matrix size
num_trials = 10;  % number of trials
eigenvalue_samples = [];  % eigenvalue samples
bin_size = .2;  % binsize

%% Experiment
for i=1:num_trials
    A = randn(n);  % random n x n matrix
    S = (A+A') / 2;  % symmetrized matrix
    eigenvalue_samples = [eigenvalue_samples; eig(S)];
end

% Normalize eigenvalues
eigenvalue_samples = eigenvalue_samples / sqrt(n/2);

%% Plot
[count, x] = hist(eigenvalue_samples, -2:bin_size:2);
cla reset
bar(x, count / (num_trials*n*bin_size), 'y');
hold on;
axis([-2.5 2.5 -.1 .5])

%% Theory
plot(x, sqrt(4-x.^2) / (2*pi), 'LineWidth', 2)
