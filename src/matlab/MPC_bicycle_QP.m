clear all; clc; clf;

%% =========================
%  Vehicle parameters
%  x = [e_y; beta; e_psi; r];
%  u = delta
%  x(k+1) = Ad x(k) + Bd u(k) + Ed kappa_ref(k)
%  MPC decision variable = Delta U
%  augmented state z = [x; u_prev]
%% =========================

m  = 1575;
Iz = 2875;
lf  = 1.2;
lr  = 1.6;
Cf = 19000;
Cr = 33000;
Vx = 15.0;

Ts = 0.05;

%% Continuous-time model
%A = [0,   Vx, 1, 0;
%     0,    0, 0, 1;
%     0,    0, -(Cf+Cr)/(m*Vx), -(Vx + (a*Cf - b*Cr)/(m*Vx));
%     0,    0, -(a*Cf - b*Cr)/(Iz*Vx), -(a^2*Cf + b^2*Cr)/(Iz*Vx)];

A = [0,   Vx, Vx, 0;
     0,  -(Cf+Cr)/(m*Vx), 0,  -1 - (lf*Cf - lr*Cr)/(m*Vx^2);
     0,   0,  0, 1;
     0,  -(lf*Cf - lr*Cr)/Iz, 0, -(lf^2*Cf + lr^2*Cr)/(Iz*Vx)];

B = [0;
     Cf/(m*Vx);
     0;
     lf*Cf/Iz];

E = [0;
     0;
    -Vx;
     0];

%% Discretization
sys_c = ss(A, B, eye(4), zeros(4,1));
sys_d = c2d(sys_c, Ts);
Ad = sys_d.A;
Bd = sys_d.B;

sys_kappa_c = ss(A, E, eye(4), zeros(4,1));
sys_kappa_d = c2d(sys_kappa_c, Ts);
Ed = sys_kappa_d.B;

nx = size(Ad,1);
nu = size(Bd,2);

%% Augmented model
% z = [x; u_prev]
A_aug = [Ad, Bd;
         zeros(nu,nx), eye(nu)];

B_aug = [Bd;
         eye(nu)];

E_aug = [Ed;
         zeros(nu,1)];

nz = size(A_aug,1);

%% MPC settings
Np = 20;
Nc = 10;

Qx = diag([250, 10, 50, 10]);   % [e_y, beta, e_psi, r]
Ru = 1;                          % penalty on steering magnitude
Rdu = 50;                        % penalty on steering increment
P  = Qx;                         % terminal weight

Qz_stage = blkdiag(Qx, Ru);
Qz_term  = blkdiag(P,  Ru);

Qbar = blkdiag(kron(eye(Np-1), Qz_stage), Qz_term);
Rbar = kron(eye(Nc), Rdu);

%% Constraints
delta_max = deg2rad(6);
delta_min = -delta_max;

deltadot_max = deg2rad(40);      % steering rate [rad/s]
deltadot_min = -deltadot_max;

du_max = deltadot_max * Ts;
du_min = deltadot_min * Ts;

%% Prediction matrices
[Sx, Su, Sd] = build_prediction_matrices(A_aug, B_aug, E_aug, Np, Nc);

% Select steering from augmented state z = [x; u_prev]
Cu = [zeros(1,nx), 1];
Cu_bar = kron(eye(Np), Cu);

%% Simulation settings
Tsim = 8.0;
Nsim = round(Tsim/Ts);

% curvature preview array
t_preview = (0:Nsim+Np)' * Ts;
kappa_profile = zeros(length(t_preview),1);

% S-curve road
idx1 = (t_preview >= 2.0) & (t_preview < 4.0);
idx2 = (t_preview >= 4.0) & (t_preview < 6.0);
kappa_profile(idx1) =  0.0025;
kappa_profile(idx2) = -0.0025;

%% Initial conditions
x = [0.5;
     0;
     deg2rad(5);
     0.0];

u_prev = 0.0;
z = [x; u_prev];

%% Logging
x_hist = zeros(nx, Nsim+1);
u_hist = zeros(1,  Nsim);
du_hist = zeros(1, Nsim);
kappa_hist = zeros(1, Nsim);
solve_time_hist = zeros(1, Nsim);
exitflag_hist = zeros(1, Nsim);

x_hist(:,1) = x;

%% Warm start initial guess
du_guess = zeros(Nc,1);

%% quadprog options
opts = optimoptions('quadprog', ...
    'Algorithm','active-set', ...
    'Display','off');
ws = optimwarmstart(zeros(Nc,1), opts);
%% Main simulation loop
for k = 1:Nsim

    % current preview of curvature over horizon
    D = kappa_profile(k:k+Np-1);

    % current augmented state
    z = [x; u_prev];

    % Hessian and gradient
    H = 2*(Su' * Qbar * Su + Rbar);
    H = (H + H')/2;

    z_free = Sx*z + Sd*D;
    f = 2*(Su' * Qbar * z_free);

    % steering magnitude constraints over prediction horizon
    U_aff = Cu_bar * z_free;
    U_mat = Cu_bar * Su;

    Aineq = [ U_mat;
             -U_mat];

    bineq = [ delta_max*ones(Np,1) - U_aff;
             -delta_min*ones(Np,1) + U_aff];

    % steering rate bounds
    lb = du_min * ones(Nc,1);
    ub = du_max * ones(Nc,1);

    % solve QP
    tic;
    %[du_star, ~, exitflag] = quadprog(H, f, Aineq, bineq, [], [], lb, ub, du_guess, opts);
    %[du_star, ~, exitflag] = quadprog(H, f, Aineq, bineq, [], [], lb, ub, zeros(Nc,1), opts);
    [ws, ~, exitflag, output] = quadprog(H, f, Aineq, bineq, [], [], lb, ub, ws);
    du_star = ws.X;
    %du_applied = du_star(1);

    solve_time_hist(k) = toc * 1000;
    exitflag_hist(k) = exitflag;

    if isempty(du_star) || exitflag <= 0
        du_applied = 0.0;
        du_star = zeros(Nc,1);
    else
        du_applied = du_star(1);
    end

    % apply control
    u = u_prev + du_applied;

    % clamp just in case
    u = min(max(u, delta_min), delta_max);

    % plant update
    kappa_now = kappa_profile(k);
    x = Ad*x + Bd*u + Ed*kappa_now;

    % log
    x_hist(:,k+1) = x;
    u_hist(k) = u;
    du_hist(k) = du_applied;
    kappa_hist(k) = kappa_now;

    % update previous input
    u_prev = u;

    % warm start shift
    du_guess = [du_star(2:end); du_star(end)];
    %du_guess = zeros(Nc,1);
end

%% =========================
%  Plot
%% =========================
t_state = 0:Ts:Tsim;
t_input = 0:Ts:Tsim-Ts;

figure;
plot(t_state, x_hist(1,:), 'LineWidth', 1.5);
grid on;
xlabel('Time [s]');
ylabel('e_y [m]');
title('Lateral error');

figure;
plot(t_state, rad2deg(x_hist(3,:)), 'LineWidth', 1.5);
grid on;
xlabel('Time [s]');
ylabel('e_\psi [deg]');
title('Heading error');

figure;
plot(t_state, rad2deg(x_hist(2,:)), 'LineWidth', 1.5); hold on;
plot(t_state, rad2deg(x_hist(4,:)), 'LineWidth', 1.5);
grid on;
xlabel('Time [s]');
ylabel('State');
legend('Beta [deg]', 'r [deg/s]');
title('Vehicle states');

figure;
stairs(t_input, rad2deg(u_hist), 'LineWidth', 1.5); hold on;
yline(rad2deg(delta_max), '--r');
yline(rad2deg(delta_min), '--r');
grid on;
xlabel('Time [s]');
ylabel('\delta [deg]');
title('Steering angle');

figure;
stairs(t_input, rad2deg(du_hist)/Ts, 'LineWidth', 1.5); hold on;
yline(rad2deg(du_max)/Ts, '--r');
yline(rad2deg(du_min)/Ts, '--r');
grid on;
xlabel('Time [s]');
ylabel('d\delta/dt [deg/s]');
title('Steering rate');

figure;
stairs(t_input, kappa_hist, 'LineWidth', 1.5);
grid on;
xlabel('Time [s]');
ylabel('\kappa_{ref} [1/m]');
title('Road curvature');

figure;
plot(t_input(4:end), solve_time_hist(4:end), 'LineWidth', 1.5);
grid on;
xlabel('Time [s]');
ylabel('Solve time [ms]');
title('QP solve time');

%% =========================
%  Summary
%% =========================
%fprintf('Average solve time = %.3f ms\n', mean(solve_time_hist));
%fprintf('Max solve time     = %.3f ms\n', max(solve_time_hist));
fprintf('Average solve time = %.3f ms\n', mean(solve_time_hist(4:end)));
fprintf('Max solve time     = %.3f ms\n', max(solve_time_hist(4:end)));
fprintf('RMS lateral error  = %.4f m\n', rms(x_hist(1,:)));
fprintf('Max |steering|     = %.3f deg\n', max(abs(rad2deg(u_hist))));
fprintf('Successful solves  = %d / %d\n', sum(exitflag_hist > 0), Nsim);

%% =========================
%  Reference path and actual vehicle path in global X-Y
%% =========================

psi_ref = zeros(1, Nsim+1);
x_ref   = zeros(1, Nsim+1);
y_ref   = zeros(1, Nsim+1);

for k = 1:Nsim
    % reference heading update
    psi_ref(k+1) = psi_ref(k) + Vx * kappa_hist(k) * Ts;

    % reference position update
    x_ref(k+1) = x_ref(k) + Vx * cos(psi_ref(k)) * Ts;
    y_ref(k+1) = y_ref(k) + Vx * sin(psi_ref(k)) * Ts;
end

% actual vehicle position from lateral error e_y
x_car = zeros(1, Nsim+1);
y_car = zeros(1, Nsim+1);

for k = 1:Nsim+1
    ey = x_hist(1,k);

    % Frenet -> global
    x_car(k) = x_ref(k) - ey * sin(psi_ref(k));
    y_car(k) = y_ref(k) + ey * cos(psi_ref(k));
end

figure;
plot(x_ref, y_ref, '--k', 'LineWidth', 2); hold on;
plot(x_car, y_car, 'b', 'LineWidth', 2);
grid on;
%axis equal;
xlabel('X [m]');
ylabel('Y [m]');
legend('Reference path', 'Vehicle path');
title('Reference vs Vehicle Trajectory');

%% =========================
%  Local function
%% =========================
function [Sx, Su, Sd] = build_prediction_matrices(A, B, E, Np, Nc)

nz = size(A,1);
nu = size(B,2);
nd = size(E,2);

Sx = zeros(nz*Np, nz);
Su = zeros(nz*Np, nu*Nc);
Sd = zeros(nz*Np, nd*Np);

for i = 1:Np
    row_idx = (i-1)*nz + (1:nz);

    Sx(row_idx,:) = A^i;

    for j = 1:min(i,Nc)
        col_u = (j-1)*nu + (1:nu);
        Su(row_idx, col_u) = A^(i-j) * B;
    end

    for j = 1:i
        col_d = (j-1)*nd + (1:nd);
        Sd(row_idx, col_d) = A^(i-j) * E;
    end
end

end