function du_guess = mlp_predict_du(X_query)
% X_query: 1x25 or 25x1
% du_guess: 10x1

persistent W1 b1 W2 b2 W3 b3 x_mean x_std y_mean y_std  %함수가 끝나도 메모리에서 지워지지 않는 변수

if isempty(W1) % 즉 W1이 persistent 변수니까, 처음에만 시행됨...! W1이 없는 경우
    S = load('mlp_warmstart_weights.mat'); %구조체 S를 꺼내오고
    W1 = S.W1;  b1 = S.b1(:); %Weight과 bias 추출
    W2 = S.W2;  b2 = S.b2(:);
    W3 = S.W3;  b3 = S.b3(:);

    x_mean = S.x_mean(:); % Normalization factor 추출
    x_std  = S.x_std(:);
    y_mean = S.y_mean(:);
    y_std  = S.y_std(:);
end

x = X_query(:);   % 25x1 

% normalize
x_n = (x - x_mean) ./ x_std;

% layer 1
h1 = W1 * x_n + b1;
h1 = max(h1, 0);   % ReLU

% layer 2
h2 = W2 * h1 + b2;
h2 = max(h2, 0);   % ReLU

% output
y_n = W3 * h2 + b3;

% denormalize
du_guess = y_n .* y_std + y_mean;
du_guess = du_guess(:);
end