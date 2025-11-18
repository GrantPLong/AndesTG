%% MATLAB Script to Generate Validation Data for Python Implementation
% This script runs the MATLAB functions on the test datasets and saves
% the results for comparison with the Python implementation.
%
% Author: Validation script for Python port
% Date: 2025

clear; close all; clc;

fprintf('\n');
fprintf('================================================================\n');
fprintf('  MATLAB Validation Data Generation for Python Implementation\n');
fprintf('================================================================\n\n');

%% Test 1: Block Uplift Inversion
fprintf('Test 1: Block Uplift Inversion\n');
fprintf('-------------------------------\n');

% Load data
load('BlockUpliftDataHighRes.mat');

% Calculate m using slope-area
[m_matlab, ~, ~, R2_matlab] = FindmSlopeArea(slope_array, area_array);
fprintf('Concavity index m = %.6f (R^2 = %.4f)\n', m_matlab, R2_matlab);

% Calculate chi
chi = CalculateChi(x, y, rec_array, area_array, m_matlab);
fprintf('Chi calculated: range [%.2f, %.2f]\n', min(chi), max(chi));

% Run inversion with different parameters
gamma = 1.0;
q = 5;

fprintf('Running inversion with gamma = %.2f, q = %d\n', gamma, q);
[Ustar_matlab, tstar_matlab, Misfit_matlab] = InvertBlockUplift(chi, z, gamma, q, 0);

fprintf('Results:\n');
fprintf('  Misfit: %.6f\n', Misfit_matlab);
fprintf('  Uplift rates (U*):\n');
for i = 1:length(Ustar_matlab)
    fprintf('    Interval %d: %.6f\n', i, Ustar_matlab(i));
end

% Test calibration
H = 500.0;  % Total uplift [m]
t_H = 2e6;  % Age [yr]
A0 = 1.0;

fprintf('\nTesting calibration with H=%.1f m, t_H=%.2e yr\n', H, t_H);
[K_matlab, U_matlab, t_matlab] = CalibrateKTotalUplift(H, t_H, Ustar_matlab, tstar_matlab, A0, m_matlab, 0);
fprintf('Erodibility K = %.6e\n', K_matlab);

% Test bootstrap (reduced iterations for speed)
fprintf('\nRunning bootstrap analysis (20 iterations)...\n');
percent_sample = 0.8;
num_iterations = 20;
rng(42);  % Set seed for reproducibility
[Ustar_mat_matlab, tstar_best_matlab] = BootstrapInvertBlockUplift(chi, z, gamma, 3, percent_sample, num_iterations, K_matlab);
fprintf('Bootstrap completed\n');

% Save results
save('matlab_validation_block_uplift.mat', ...
    'm_matlab', 'R2_matlab', 'chi', ...
    'Ustar_matlab', 'tstar_matlab', 'Misfit_matlab', ...
    'K_matlab', 'U_matlab', 't_matlab', ...
    'Ustar_mat_matlab', 'tstar_best_matlab', ...
    'gamma', 'q', 'H', 't_H', 'A0');

fprintf('Results saved to matlab_validation_block_uplift.mat\n\n');

%% Test 2: Parabola Inversion
fprintf('Test 2: Parabola Inversion\n');
fprintf('---------------------------\n');

% Load parabola data
load('ParabolaDataHighRes.mat');

% Calculate m and chi
[m_parab, ~, ~, R2_parab] = FindmSlopeArea(slope_array, area_array);
fprintf('Concavity index m = %.6f (R^2 = %.4f)\n', m_parab, R2_parab);

chi_parab = CalculateChi(x, y, rec_array, area_array, m_parab);
fprintf('Chi calculated: range [%.2f, %.2f]\n', min(chi_parab), max(chi_parab));

% Run parabola inversion
gamma_parab = 1.0;
q_parab = 3;
K_parab = 1.0;

fprintf('Running parabola inversion with gamma = %.2f, q = %d\n', gamma_parab, q_parab);
[Up_matlab, tstar_parab_matlab, Misfit_parab_matlab] = InvertParabola(chi_parab, z, x, rec_array, gamma_parab, q_parab, K_parab, 0);

fprintf('Results:\n');
fprintf('  Misfit: %.6f\n', Misfit_parab_matlab);
fprintf('  Parabola coefficients:\n');
for i = 1:q_parab
    a = Up_matlab((i-1)*3 + 1);
    b = Up_matlab((i-1)*3 + 2);
    c = Up_matlab((i-1)*3 + 3);
    fprintf('    Interval %d: a=%.6e, b=%.6e, c=%.6f\n', i, a, b, c);
end

% Save results
save('matlab_validation_parabola.mat', ...
    'm_parab', 'R2_parab', 'chi_parab', ...
    'Up_matlab', 'tstar_parab_matlab', 'Misfit_parab_matlab', ...
    'gamma_parab', 'q_parab', 'K_parab');

fprintf('Results saved to matlab_validation_parabola.mat\n\n');

%% Test 3: Different Gamma Values (L-Curve)
fprintf('Test 3: L-Curve Analysis\n');
fprintf('-------------------------\n');

% Use block uplift data
load('BlockUpliftDataHighRes.mat');
chi = CalculateChi(x, y, rec_array, area_array, m_matlab);

fprintf('Testing 10 gamma values...\n');
InvertWithDifferentGamma(chi, z, 5);
title('MATLAB L-Curve');

fprintf('L-curve plot generated\n\n');

%% Summary
fprintf('================================================================\n');
fprintf('  VALIDATION DATA GENERATION COMPLETE\n');
fprintf('================================================================\n');
fprintf('\nGenerated files:\n');
fprintf('  - matlab_validation_block_uplift.mat\n');
fprintf('  - matlab_validation_parabola.mat\n');
fprintf('\nThese files can be compared with Python results using the\n');
fprintf('Python script: python/tests/test_matlab_equivalence.py\n\n');
