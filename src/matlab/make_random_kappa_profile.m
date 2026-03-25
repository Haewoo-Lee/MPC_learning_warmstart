function kappa_profile = make_random_kappa_profile(Nsim, Np)

L = Nsim + Np - 1;
kappa_profile = zeros(L,1);

nSeg = randi([2,4]); %2~4중 random
bp = sort(randperm(L-20, nSeg-1) + 10); %1 ~ L-20 까지 안겹치게 nSeg -1개 추출 % 오름차순으로 정렬
idx = [1; bp(:); L+1]; 

val = 0.003 * (2*rand(nSeg,1) - 1); %rand: 0~1 사이 random %2rand -1 : -1 ~ 1 까지 nSeg개 추출
val(1) = 0;

for i = 1:nSeg
    kappa_profile(idx(i):idx(i+1)-1) = val(i);
end

kappa_profile = movmean(kappa_profile, 5); %moving vaverage filter 앞 2개 뒤 2개 자신값 해서 평균~ 
kappa_profile(1:min(20,L)) = 0; %시나리오가 짧으면 그냥 0으로 취급

end