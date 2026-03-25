function out = run_mpc_case(x0, kappa_profile, p)

m  = p.m;
Iz = p.Iz;
lf = p.lf;
lr = p.lr;
Cf = p.Cf;
Cr = p.Cr;
Vx = p.Vx;

Ts = p.Ts;
Np = p.Np;
Nc = p.Nc;
Tsim = p.Tsim;

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

if length(kappa_profile) < Nsim + Np - 1
    error('kappa_profile length is too short.');
end

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
         zeros(nu,nx), eye(nu)]; %% states [x(k), u(k-1)]

B_aug = [Bd;
         eye(nu)];

E_aug = [Ed;
         zeros(nu,1)];

nz = size(A_aug,1);

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

x_hist         = zeros(nx, Nsim+1);
x_pre_hist     = zeros(nx, Nsim);
u_prev_hist    = zeros(1, Nsim);
D_hist         = zeros(Np, Nsim);
du_star_hist   = zeros(Nc, Nsim);
u_hist         = zeros(1, Nsim);
du_hist        = zeros(1, Nsim);
kappa_hist     = zeros(1, Nsim);
solve_time_hist= zeros(1, Nsim);
iter_hist      = zeros(1, Nsim);
exitflag_hist  = zeros(1, Nsim);

x_hist(:,1) = x;

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

    tic;
    [du_star, ~, exitflag, output] = quadprog(H, f, Aineq, bineq, [], [], lb, ub, zeros(Nc,1), opts);
    solve_time_hist(k) = toc * 1000;
    exitflag_hist(k) = exitflag;

    if isempty(du_star) || exitflag <= 0
        du_star = zeros(Nc,1);
        du_applied = 0.0;
    else
        du_applied = du_star(1);
    end

    if isfield(output, 'iterations')
        iter_hist(k) = output.iterations;
    else
        iter_hist(k) = NaN;
    end

    x_pre_hist(:,k)   = x;
    u_prev_hist(k)    = u_prev;
    D_hist(:,k)       = D;
    du_star_hist(:,k) = du_star;

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

out.x_hist          = x_hist;
out.x_pre_hist      = x_pre_hist;
out.u_prev_hist     = u_prev_hist;
out.D_hist          = D_hist;
out.du_star_hist    = du_star_hist;
out.u_hist          = u_hist;
out.du_hist         = du_hist;
out.kappa_hist      = kappa_hist;
out.solve_time_hist = solve_time_hist;
out.iter_hist       = iter_hist;
out.exitflag_hist   = exitflag_hist;

end