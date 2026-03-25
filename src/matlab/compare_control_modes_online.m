clear; clc; close all;

%% =========================
% Settings
%% =========================
%rng(10);

p.m  = 1575;
p.Iz = 2875;
p.lf = 1.2;
p.lr = 1.6;
p.Cf = 19000;
p.Cr = 33000;
p.Vx = 30.0;
%p.Vx = 15.0;


p.Ts = 0.05;
p.Tsim = 8.0;

p.Np = 20;
p.Nc = 10;

p.Qx = diag([250, 10, 50, 10]);
p.Ru = 1;
p.Rdu = 50;

p.delta_max = deg2rad(6);
p.deltadot_max = deg2rad(40);

num_test_cases = 100;   % 원하는 만큼 바꿔도 됨
Nsim = round(p.Tsim / p.Ts);

%% =========================
% Preallocate results
%% =========================
results_mode1 = repmat(empty_result_struct(), num_test_cases, 1);
results_mode2 = repmat(empty_result_struct(), num_test_cases, 1);
results_mode3 = repmat(empty_result_struct(), num_test_cases, 1);
results_mode4 = repmat(empty_result_struct(), num_test_cases, 1);

example_out = cell(4,1);
example_case_id = 10;

%% =========================
% Main test loop
%% =========================
for i = 1:num_test_cases

    x0 = [ -1.0 + 2.0*rand;
           deg2rad(-3 + 6*rand);
           deg2rad(-8 + 16*rand);
           deg2rad(-10 + 20*rand) ];

    kappa_profile = make_random_kappa_profile(Nsim, p.Np);

    out1 = run_case_with_mode(x0, kappa_profile, p, 1);
    out2 = run_case_with_mode(x0, kappa_profile, p, 2);
    out3 = run_case_with_mode(x0, kappa_profile, p, 3);
    out4 = run_case_with_mode(x0, kappa_profile, p, 4);

    results_mode1(i) = summarize_case(out1, i);
    results_mode2(i) = summarize_case(out2, i);
    results_mode3(i) = summarize_case(out3, i);
    results_mode4(i) = summarize_case(out4, i);

    if i == example_case_id
        example_out{1} = out1;
        example_out{2} = out2;
        example_out{3} = out3;
        example_out{4} = out4;
    end

    if mod(i,10) == 0 || i == num_test_cases
        fprintf('Done %d / %d cases\n', i, num_test_cases);
    end
end

%% =========================
% Print summary
%% =========================
fprintf('\n==============================\n');
fprintf('Mode 1: Cold-start MPC\n');
print_summary(results_mode1);

fprintf('\n==============================\n');
fprintf('Mode 2: Shifted warm-start MPC\n');
print_summary(results_mode2);

fprintf('\n==============================\n');
fprintf('Mode 3: NN warm-start MPC\n');
print_summary(results_mode3);

fprintf('\n==============================\n');
fprintf('Mode 4: Direct NN control\n');
print_summary(results_mode4);

%% =========================
% Plot one representative case
%% =========================
t_state = 0:p.Ts:p.Tsim;
t_input = 0:p.Ts:p.Tsim-p.Ts;

figure;
plot(t_state, example_out{1}.x_hist(1,:), 'LineWidth', 1.5); hold on;
plot(t_state, example_out{2}.x_hist(1,:), 'LineWidth', 1.5);
plot(t_state, example_out{3}.x_hist(1,:), 'LineWidth', 1.5);
plot(t_state, example_out{4}.x_hist(1,:), 'LineWidth', 1.5);
grid on;
xlabel('Time [s]');
ylabel('e_y [m]');
legend('Cold MPC','Shift MPC','NN warm MPC','Direct NN');
title('Representative case: lateral error');

figure;
stairs(t_input, rad2deg(example_out{1}.u_hist), 'LineWidth', 1.5); hold on;
stairs(t_input, rad2deg(example_out{2}.u_hist), 'LineWidth', 1.5);
stairs(t_input, rad2deg(example_out{3}.u_hist), 'LineWidth', 1.5);
stairs(t_input, rad2deg(example_out{4}.u_hist), 'LineWidth', 1.5);
grid on;
xlabel('Time [s]');
ylabel('\delta [deg]');
legend('Cold MPC','Shift MPC','NN warm MPC','Direct NN');
title('Representative case: steering');

figure;
plot(t_input, example_out{1}.calc_time_ms, 'LineWidth', 1.5); hold on;
plot(t_input, example_out{2}.calc_time_ms, 'LineWidth', 1.5);
plot(t_input, example_out{3}.calc_time_ms, 'LineWidth', 1.5);
plot(t_input, example_out{4}.calc_time_ms, 'LineWidth', 1.5);
grid on;
xlabel('Time [s]');
ylabel('Calculation time [ms]');
legend('Cold MPC','Shift MPC','NN warm MPC','Direct NN');
title('Representative case: calculation time');

figure;
hold on;

for mode_idx = 1:4
    x_hist = example_out{mode_idx}.x_hist;
    kappa_hist = example_out{mode_idx}.kappa_hist;

    psi_ref = zeros(1, Nsim+1);
    x_ref   = zeros(1, Nsim+1);
    y_ref   = zeros(1, Nsim+1);

    for k = 1:Nsim
        psi_ref(k+1) = psi_ref(k) + p.Vx * kappa_hist(k) * p.Ts;
        x_ref(k+1) = x_ref(k) + p.Vx * cos(psi_ref(k)) * p.Ts;
        y_ref(k+1) = y_ref(k) + p.Vx * sin(psi_ref(k)) * p.Ts;
    end

    x_car = zeros(1, Nsim+1);
    y_car = zeros(1, Nsim+1);

    for k = 1:Nsim+1
        ey = x_hist(1,k);
        x_car(k) = x_ref(k) - ey * sin(psi_ref(k));
        y_car(k) = y_ref(k) + ey * cos(psi_ref(k));
    end

    if mode_idx == 1
        plot(x_ref, y_ref, '--k', 'LineWidth', 2);
    end

    plot(x_car, y_car, 'LineWidth', 1.8);
end

grid on;
xlabel('X [m]');
ylabel('Y [m]');
legend('Reference path', 'Cold MPC', 'Shift MPC', 'NN warm MPC', 'Direct NN');
title('Representative case: Reference vs Vehicle Trajectory');

%% =========================
% Save results
%% =========================
save('control_mode_comparison_results.mat', ...
    'results_mode1', 'results_mode2', 'results_mode3', 'results_mode4', 'p');

fprintf('\nSaved: control_mode_comparison_results.mat\n');

%% =========================
% Local functions
%% =========================
function out = run_case_with_mode(x0, kappa_profile, p, mode)

m  = p.m;
Iz = p.Iz;
lf = p.lf;
lr = p.lr;
Cf = p.Cf;
Cr = p.Cr;
Vx = p.Vx;

Ts   = p.Ts;
Tsim = p.Tsim;
Np   = p.Np;
Nc   = p.Nc;

Qx  = p.Qx;
Ru  = p.Ru;
Rdu = p.Rdu;
P   = Qx;

delta_max = p.delta_max;
delta_min = -delta_max;

deltadot_max = p.deltadot_max;
deltadot_min = -deltadot_max;

du_max = deltadot_max * Ts;
du_min = deltadot_min * Ts;

Nsim = round(Tsim / Ts);

A = [0,  Vx, Vx, 0;
     0, -(Cf+Cr)/(m*Vx), 0, -1 - (lf*Cf - lr*Cr)/(m*Vx^2);
     0, 0, 0, 1;
     0, -(lf*Cf - lr*Cr)/Iz, 0, -(lf^2*Cf + lr^2*Cr)/(Iz*Vx)];

B = [0;
     Cf/(m*Vx);
     0;
     lf*Cf/Iz];

E = [0;
     0;
    -Vx;
     0];

sys_c = ss(A, B, eye(4), zeros(4,1));
sys_d = c2d(sys_c, Ts);
Ad = sys_d.A;
Bd = sys_d.B;

sys_kappa_c = ss(A, E, eye(4), zeros(4,1));
sys_kappa_d = c2d(sys_kappa_c, Ts);
Ed = sys_kappa_d.B;

nx = size(Ad,1);
nu = size(Bd,2);

A_aug = [Ad, Bd;
         zeros(nu,nx), eye(nu)];

B_aug = [Bd;
         eye(nu)];

E_aug = [Ed;
         zeros(nu,1)];

Qz_stage = blkdiag(Qx, Ru);
Qz_term  = blkdiag(P,  Ru);

Qbar = blkdiag(kron(eye(Np-1), Qz_stage), Qz_term);
Rbar = kron(eye(Nc), Rdu);

[Sx, Su, Sd] = build_prediction_matrices(A_aug, B_aug, E_aug, Np, Nc);

Cu = [zeros(1,nx), 1];
Cu_bar = kron(eye(Np), Cu);

H = 2*(Su' * Qbar * Su + Rbar);
H = (H + H')/2;

opts = optimoptions('quadprog', ...
    'Algorithm','active-set', ...
    'Display','off');

x = x0(:);
u_prev = 0.0;

x_hist = zeros(nx, Nsim+1);
u_hist = zeros(1, Nsim);
du_hist = zeros(1, Nsim);
calc_time_ms = zeros(1, Nsim);
iter_hist = nan(1, Nsim);
exitflag_hist = nan(1, Nsim);
kappa_hist = zeros(1, Nsim);

x_hist(:,1) = x;

ws = optimwarmstart(zeros(Nc,1), opts);
du_guess_shift = zeros(Nc,1);

for k = 1:Nsim

    D = kappa_profile(k:k+Np-1);
    z = [x; u_prev];

    z_free = Sx*z + Sd*D;
    f = 2*(Su' * Qbar * z_free);

    U_aff = Cu_bar * z_free;
    U_mat = Cu_bar * Su;

    Aineq = [ U_mat;
             -U_mat];

    bineq = [ delta_max*ones(Np,1) - U_aff;
             -delta_min*ones(Np,1) + U_aff];

    lb = du_min * ones(Nc,1);
    ub = du_max * ones(Nc,1);

    switch mode
        case 1
            tic;
            [du_star, ~, exitflag, output] = quadprog(H, f, Aineq, bineq, [], [], lb, ub, zeros(Nc,1), opts);
            calc_time_ms(k) = toc * 1000;

            if isempty(du_star) || exitflag <= 0
                du_star = zeros(Nc,1);
                du_applied = 0.0;
            else
                du_applied = du_star(1);
            end

            if isfield(output, 'iterations')
                iter_hist(k) = output.iterations;
            end
            exitflag_hist(k) = exitflag;

        case 2
            tic;            
            [ws, ~, exitflag, output] = quadprog(H, f, Aineq, bineq, [], [], lb, ub, ws); du_star = ws.X; %optimum warm start 
            %[du_star, ~, exitflag, output] = quadprog(H, f, Aineq, bineq, [], [], lb, ub, du_guess_shift, opts);  %Shifted warm-start MPC
            calc_time_ms(k) = toc * 1000;   
            
        
            if isempty(du_star) || exitflag <= 0
                du_star = zeros(Nc,1);
                du_applied = 0.0;
        
                %optimum wart start시 켜놔야함 실패 시 warm start object를 다시 초기화, x0만 있을때는 꺼놔야함.
                ws = optimwarmstart(zeros(Nc,1), opts);
            else
                du_applied = du_star(1);
            end
        
            if isfield(output, 'iterations')
                iter_hist(k) = output.iterations;
            end
            exitflag_hist(k) = exitflag;
            du_guess_shift = [du_star(2:end); du_star(end)];

        case 3
            X_query = [x(:).', u_prev, D(:).'];

            tic;
            du_guess_nn = mlp_predict_du(X_query);
            du_guess_nn = min(max(du_guess_nn, lb), ub);
            [du_star, ~, exitflag, output] = quadprog(H, f, Aineq, bineq, [], [], lb, ub, du_guess_nn, opts);
            calc_time_ms(k) = toc * 1000;

            if isempty(du_star) || exitflag <= 0
                du_star = zeros(Nc,1);
                du_applied = 0.0;
            else
                du_applied = du_star(1);
            end

            if isfield(output, 'iterations')
                iter_hist(k) = output.iterations;
            end
            exitflag_hist(k) = exitflag;

        case 4
            X_query = [x(:).', u_prev, D(:).'];

            tic;
            du_guess_nn = mlp_predict_du(X_query);
            du_applied = du_guess_nn(1);
            du_applied = min(max(du_applied, du_min), du_max);
            calc_time_ms(k) = toc * 1000;

            iter_hist(k) = NaN;
            exitflag_hist(k) = 1;

        otherwise
            error('Unknown mode');
    end

    u = u_prev + du_applied;
    u = min(max(u, delta_min), delta_max);

    kappa_now = kappa_profile(k);
    x = Ad*x + Bd*u + Ed*kappa_now;

    x_hist(:,k+1) = x;
    u_hist(k) = u;
    du_hist(k) = du_applied;
    kappa_hist(k) = kappa_now;

    u_prev = u;
end

out.x_hist = x_hist;
out.u_hist = u_hist;
out.du_hist = du_hist;
out.calc_time_ms = calc_time_ms;
out.iter_hist = iter_hist;
out.exitflag_hist = exitflag_hist;
out.kappa_hist = kappa_hist;

end

function S = summarize_case(out, cid)
S.case_id = cid;
S.rms_ey = rms(out.x_hist(1,:));
S.rms_epsi_deg = rms(rad2deg(out.x_hist(3,:)));
S.max_abs_delta_deg = max(abs(rad2deg(out.u_hist)));
S.mean_time_ms = mean(out.calc_time_ms);
S.p95_time_ms = prctile(out.calc_time_ms, 95);
S.p99_time_ms = prctile(out.calc_time_ms, 99);
S.max_time_ms = max(out.calc_time_ms);
S.mean_iter = mean(out.iter_hist(~isnan(out.iter_hist)));
S.max_iter = max(out.iter_hist(~isnan(out.iter_hist)));
S.success_rate = mean(out.exitflag_hist > 0);
end

function print_summary(T)

rms_ey = [T.rms_ey];
rms_epsi_deg = [T.rms_epsi_deg];
max_abs_delta_deg = [T.max_abs_delta_deg];
mean_time_ms = [T.mean_time_ms];
p95_time_ms = [T.p95_time_ms];
p99_time_ms = [T.p99_time_ms];
max_time_ms = [T.max_time_ms];
success_rate = [T.success_rate];

valid_iter = [T.max_iter];
valid_iter = valid_iter(~isnan(valid_iter));
all_iter = [T.mean_iter];
all_iter = all_iter(~isnan(all_iter));

fprintf('Cases evaluated      = %d\n', length(T));
fprintf('Mean RMS e_y         = %.4f m\n', mean(rms_ey));
fprintf('Mean RMS e_psi       = %.4f deg\n', mean(rms_epsi_deg));
fprintf('Mean max |delta|     = %.4f deg\n', mean(max_abs_delta_deg));
fprintf('Mean calc time       = %.6f ms\n', mean(mean_time_ms));
fprintf('Mean p95 time        = %.6f ms\n', mean(p95_time_ms));
fprintf('Mean p99 time        = %.6f ms\n', mean(p99_time_ms));
fprintf('Worst max time       = %.6f ms\n', max(max_time_ms));
fprintf('Mean success rate    = %.2f %%\n', 100*mean(success_rate));

if ~isempty(all_iter)
    fprintf('Mean iterations      = %.4f\n', mean(all_iter));
else
    fprintf('Mean iterations      = NaN (not applicable)\n');
end

if ~isempty(valid_iter)
    fprintf('Max iterations      = %.4f\n', max(valid_iter));
else
    fprintf('Max iterations      = NaN (not applicable)\n');
end


end

function S = empty_result_struct()
S.case_id = NaN;
S.rms_ey = NaN;
S.rms_epsi_deg = NaN;
S.max_abs_delta_deg = NaN;
S.mean_time_ms = NaN;
S.p95_time_ms = NaN;
S.p99_time_ms = NaN;
S.max_time_ms = NaN;
S.mean_iter = NaN;
S.max_iter = NaN;
S.success_rate = NaN;
end
