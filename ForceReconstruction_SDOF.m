% Efficient dynamic modal load reconstruction using physics-informed Gaussian processes based on frequency-sparse Fourier basis functions
% Tondo, G.R., Kavrakov, I., Morgenthal, G.
% Mechanical Systems and Signal Processing 225, pp. 112295 (2025)
% DOI: https://doi.org/10.1016/j.ymssp.2024.112295

%% Model implementation example
clc; clear; close all; rng(0);

%% Generate synthetic structural responses
% SDOF Model properties
m = 1;
zeta = 0.05;
f = 1;
dt = 0.05;
t = (0:dt:30);

% True force
Nfreqs = 5;
f_force = 2*rand(Nfreqs,1) + 0.25;
a_force = 0.5*randn(1,Nfreqs);
F = a_force*sin(2*pi*f_force*t);

% Get SDOF response
SDOFModel = Functions.SDOF('m', m, 'f', f, 'zeta', zeta, 'dt', dt, 'SNR', Inf);     
res = SDOFModel.integrate(F, 0);

% Add noise to response
SNR = 20;
res = res + bsxfun(@times, randn(size(res)), max(abs(res))/SNR);


%% Start the PIGP Model
% Select training points
trainIDXGP = floor(linspace(1, length(F), 200));

% Limits to GP model
% Maximum and minimum frequencies to be considered
freqMin = 0.25;
freqMax = 3;

% Spectrum filtering: only peaks above [mean * filter(1) + stdev * filter(2)]
freqFilter = [0.2, 0];

% Start GP Model
GP = Functions.GPModel(SDOFModel.m, SDOFModel.c, SDOFModel.k, 'jitter', 1e-15, 'verbose', 2, ...
    'freqMin', freqMin, 'freqMax', freqMax, 'freqFilter', freqFilter); 

% Add training data
GP.addTrainingData(t(trainIDXGP), res(trainIDXGP,1), 'u', 1);
GP.addTrainingData(t(trainIDXGP), res(trainIDXGP,2), 'v', 1);
GP.addTrainingData(t(trainIDXGP), res(trainIDXGP,3), 'a', 1);

% Train model
GP.train();

% Predict forces
[~, ~, ResGP] = GP.predict(t);

%% Plot results
figure; 

% SDOF displacements
subplot(2,2,1); hold on; grid on; box on;
plot(t, res(:,1), '-k')
xlabel('$t$ [s]', 'interpreter', 'latex')
ylabel('$u$ [m]', 'interpreter', 'latex')
ytickformat('%1.2f');
ylim([-0.06, 0.06]); yticks(-0.06:0.03:0.06)
xticks(0:5:30);

% SDOF accelerations
subplot(2,2,2); hold on; grid on; box on;
plot(t, res(:,3), '-k')
% xlim([30, 40]);
xlabel('$t$ [s]', 'interpreter', 'latex')
ylabel('$\ddot{u}$ [m/s$^2$]', 'interpreter', 'latex')
ytickformat('%1.0f');
ylim([-6, 6]); yticks(-6:3:6)
xticks(0:5:30);

% Force comparison: time domain
subplot(2,2,3); hold on; grid on; box on;
mult = 2.54;
fill([t'; flip(t')], [ResGP.MuF+mult*ResGP.SigF, flip(ResGP.MuF-mult*ResGP.SigF)], [1,1,1]*0.3, ...
        'edgecolor', 'none', 'facecolor', 'red', 'facealpha', 0.3, 'displayname', '$\sigma_{q,\mathrm{GP}}$');
plot(t, F, '-k', 'DisplayName', '$q_{\mathrm{true}}$')
plot(t, ResGP.MuF, '-r', 'DisplayName', '\mu{q,\mathrm{GP}}$')
xticks(0:5:30);
ylim([-5, 5]); yticks(-5:2.5:5)
ytickformat('%1.1f');
xlabel('$t$ [s]', 'interpreter', 'latex')
ylabel('$q$ [N]', 'interpreter', 'latex')

% Force comparison: frequency domain
[psdF, fVecF] = Functions.PSD(F, dt);
[psdGP, fVecGP] = Functions.PSD(ResGP.MuF, dt);
subplot(2,2,4); hold on; grid on; box on;
fill([[freqMin, freqMax]'; flip([freqMin, freqMax]')], [[1e-9, 1e-9], [1e3, 1e3]], [1,1,1]*0.3, ...
        'edgecolor', 'none', 'facecolor', 'blue', 'facealpha', 0.1, 'displayname', 'Valid $f$ range');
plot(fVecF, psdF.*fVecF./var(F), '-k', 'linewidth', 1, 'DisplayName', '$q_{\mathrm{true}}$');
plot(fVecGP, psdGP.*fVecGP./var(ResGP.MuF), '-r', 'DisplayName', '$\mu_{q,\mathrm{GP}}$');
set(gca, 'xscale', 'log', 'yscale', 'log')
ylim([1e-9, 1e3]); xlim([1e-1, 1e1])
xlabel('$f$ [Hz]', 'interpreter', 'latex')
ylabel('$f S_{qq}/\sigma_q^2$ [-]', 'interpreter', 'latex')
legend('location', 'best', 'interpreter', 'latex')
