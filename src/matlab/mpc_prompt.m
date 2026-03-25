clear; clc; close all;
rng(1);

p.m  = 1575;
p.Iz = 2875;
p.lf = 1.2;
p.lr = 1.6;
p.Cf = 19000;
p.Cr = 33000;
p.Vx = 15.0;

p.Ts = 0.05;
p.Tsim = 8.0;

p.Np = 20;
p.Nc = 10;

p.Qx = diag([250, 10, 50, 10]);
p.Ru = 1;
p.Rdu = 50;

p.delta_max = deg2rad(6);
p.deltadot_max = deg2rad(40);

num_cases = 500;
Nsim = round(p.Tsim / p.Ts);

nFeature = 4 + 1 + p.Np; %x_k , u_k-1 , kappa_k ~ kappa_k_np-1
nTarget  = p.Nc; %u_k ~ u_k+Nc-1
nTotal   = num_cases * Nsim; 

X_data        = zeros(nTotal, nFeature);
Y_data        = zeros(nTotal, nTarget);
solve_time_ms = zeros(nTotal, 1);
iter_data     = zeros(nTotal, 1);
exitflag_data = zeros(nTotal, 1);
case_id       = zeros(nTotal, 1);
step_id       = zeros(nTotal, 1);

row = 1;

for i = 1:num_cases

    x0 = [ -1.0 + 2.0*rand;
           deg2rad(-3 + 6*rand);
           deg2rad(-8 + 16*rand);
           deg2rad(-10 + 20*rand) ];

    kappa_profile = make_random_kappa_profile(Nsim, p.Np);

    out = run_mpc_case(x0, kappa_profile, p);

    for k = 1:Nsim
        X_data(row,:) = [out.x_pre_hist(:,k).', out.u_prev_hist(k), out.D_hist(:,k).'];
        Y_data(row,:) = out.du_star_hist(:,k).';

        solve_time_ms(row) = out.solve_time_hist(k);
        iter_data(row)     = out.iter_hist(k);
        exitflag_data(row) = out.exitflag_hist(k);
        case_id(row)       = i;
        step_id(row)       = k;

        row = row + 1;
    end

    if mod(i,50) == 0
        fprintf('case %d / %d done\n', i, num_cases);
    end
end

save('dataset_mpc_warmstart.mat', ...
    'X_data', 'Y_data', ...
    'solve_time_ms', 'iter_data', 'exitflag_data', ...
    'case_id', 'step_id', 'p', '-v7.3');

fprintf('saved: dataset_mpc_warmstart.mat\n');
fprintf('X size = [%d %d]\n', size(X_data,1), size(X_data,2));
fprintf('Y size = [%d %d]\n', size(Y_data,1), size(Y_data,2));
fprintf('mean solve time = %.3f ms\n', mean(solve_time_ms));
fprintf('max solve time  = %.3f ms\n', max(solve_time_ms));

save('dataset_mpc_warmstart_v7.mat', ...
    'X_data', 'Y_data', ...
    'solve_time_ms', 'iter_data', 'exitflag_data', ...
    'case_id', 'step_id', 'p', '-v7');
%%
feature_names = ["e_y","beta","e_psi","r","u_prev"];
for i = 1:p.Np
    feature_names(end+1) = "kappa_" + i;
end

target_names = strings(1,p.Nc);
for i = 1:p.Nc
    target_names(i) = "du_" + i;
end

save('dataset_mpc_warmstart_v7.mat', ...
    'X_data', 'Y_data', ...
    'solve_time_ms', 'iter_data', 'exitflag_data', ...
    'case_id', 'step_id', 'p', ...
    'feature_names', 'target_names', '-v7');
