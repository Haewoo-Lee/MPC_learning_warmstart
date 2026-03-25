import numpy as np
import scipy.io as sio
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# 1. Load MATLAB dataset
# -----------------------------
mat = sio.loadmat("dataset_mpc_warmstart_v7.mat")

X_data = mat["X_data"].astype(np.float32)   # shape: [N, 25] #float32는 64비트를 32비트로 바꾸는 걸로 학습 속도를 높임
Y_data = mat["Y_data"].astype(np.float32)   # shape: [N, 10]
case_id = mat["case_id"].squeeze().astype(np.int64) # matlab size n * 1 로 2차원화 되어있는데 squeeze()를 통해 [N,] 로 1차원으로 만들기
step_id = mat["step_id"].squeeze().astype(np.int64) # int64가 정수표현더 잘되게함

print("X_data shape:", X_data.shape)
print("Y_data shape:", Y_data.shape)


# -----------------------------
# 2. Split by case_id
# -----------------------------
all_cases = np.unique(case_id) #case_id 뽑아내기 case_id = [1,1,1,...,2,2,2,..,3,3,3,...,160,160,...] 이렇게 되어있는데  all_cases = [1,2,3,4,...,160] 으로 줄여줌
rng = np.random.default_rng(1) #시드 번호 지정
rng.shuffle(all_cases) #막 섞기

n_case = len(all_cases) #length of all_cases 160인가 나오지 않을까 항의 갯수임
n_train = int(0.7 * n_case) #70%정도의 데이터를 train으로 쓰겠다. 정수로 내림하기 위함
n_val = int(0.15 * n_case) #15%정도의 데이터를 val로 쓰겠다.

train_cases = all_cases[:n_train]
val_cases = all_cases[n_train:n_train + n_val] # 최고의 state인지 검증용
test_cases = all_cases[n_train + n_val:] #나머지를 test로 쓰겠다

train_mask = np.isin(case_id, train_cases) #case id 들 중 train_case에 포함되면 true 출력 아니면 False 출력
val_mask = np.isin(case_id, val_cases) 
test_mask = np.isin(case_id, test_cases)
 
X_train = X_data[train_mask] #train mask의 데이터만 뽑아옴; 오호
Y_train = Y_data[train_mask]

X_val = X_data[val_mask]
Y_val = Y_data[val_mask]

X_test = X_data[test_mask]
Y_test = Y_data[test_mask]

print("Train:", X_train.shape, Y_train.shape)
print("Val  :", X_val.shape, Y_val.shape)
print("Test :", X_test.shape, Y_test.shape)


# -----------------------------
# 3. Normalize using training set
# -----------------------------
x_mean = X_train.mean(axis=0, keepdims=True) # shape: [N, 25] -> axis 0 : 세로방향 mean and keepdimstrue 떄문에 [25] 1차원이 아닌 shape : [1,25]가 됨
x_std = X_train.std(axis=0, keepdims=True) + 1e-8 # 마찬가지로 [1,25]가 되겠죠?

y_mean = Y_train.mean(axis=0, keepdims=True)
y_std = Y_train.std(axis=0, keepdims=True) + 1e-8

X_train_n = (X_train - x_mean) / x_std # 넘파이가 똑똑하게 $[1, 25]$짜리 평균값을 모든 행($n_{train}$개)에 똑같이 복사해서 빼줌
X_val_n = (X_val - x_mean) / x_std #나누는 것도 마찬가지 shape  : [N_val*25]
X_test_n = (X_test - x_mean) / x_std

Y_train_n = (Y_train - y_mean) / y_std
Y_val_n = (Y_val - y_mean) / y_std
Y_test_n = (Y_test - y_mean) / y_std


# -----------------------------
# 4. PyTorch dataset
# -----------------------------
class MPCDataset(Dataset): #class는 matlab function이랑 다른듯 function은 망치며 class는 가방이래.. 
    def __init__(self, X, Y): # class 호출될때 자동으로 실행되는 거인듯
        self.X = torch.from_numpy(X) #ndarray를 파이토치 tensor로 변환 pytorch는 tensor만 읽는대 이를 self의 X칸에 저장
        self.Y = torch.from_numpy(Y) 

    def __len__(self): #__len__과 __getitem__은 DataLoader의 필수 function
        return self.X.shape[0] #만약 데이터 갯수가 N*25개 때: shape[0] 은 row 갯수임 data sample 갯수를 의미

    def __getitem__(self, idx): 
        return self.X[idx], self.Y[idx] #return 이 두개여서 tuple 바귄안에 (Tensor,Tensor) 로 담김 이때 크기는 (25,) 와 (10,) 1차원 tensor


train_ds = MPCDataset(X_train_n, Y_train_n) #Dataset 호출하고
val_ds = MPCDataset(X_val_n, Y_val_n)
test_ds = MPCDataset(X_test_n, Y_test_n)

train_loader = DataLoader(train_ds, batch_size=512, shuffle=True) #Dataloader 객체 만든것 train_ds의 데이터를 쓸거고, batch_size 512개의 (X,Y) 를 가져다 쓸거임 이때 shuffle 후 batch_size만큼 가져감. 
val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False) #남은게 batch_size 보다 작다면 그대로 가져오는 듯
test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False)


# -----------------------------
# 5. MLP model
# -----------------------------
class MLPWarmStart(nn.Module):
    def __init__(self, in_dim=25, out_dim=10): #기본값임 ; MLPwarmStart(26,10) 이렇게 해도 돌아가긴할듯
        super().__init__() #pytorch가 만든 기본 모델 틀을 그대로 물려받는다는 의미
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, x):
        return self.net(x)


device = "cuda" if torch.cuda.is_available() else "cpu"
model = MLPWarmStart(in_dim=X_train.shape[1], out_dim=Y_train.shape[1]).to(device) # NN 모델 넣기

criterion = nn.MSELoss() #채점 기준. (예측값 - 실제값)^2의 평균을 냄 
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) #최적화 도구 momentum를 이용해줌
#initial matrix는 kaiming initialization 과 같은 방식으로 랜덤한 숫자로 넣음 zeros로 하면 학습이 전혀 안된대. 
# 그 후에 알고있는 gradient 값으로 기울기를 계산 기울기와 error를 통해 최적화 진행
# 잘생각해보면 y = w*x 이렇게 되어있으면 w를 업데이트 해야하는데 Loss= (y - 정답)^2일때 
# dLoss/dw = dLoss/dy *dy/dx = 2(y - 정답) * x 이런느낌으로 데이터, 에러등이 다 들어감!

# -----------------------------
# 6. Train loop
# -----------------------------
def evaluate(loader):
    model.eval() #시험모드다! 학습이 아닌 실력확인과정, 모든 뉴런 100% 가동. 
    total_loss = 0.0 
    count = 0
    with torch.no_grad(): # 기울기 계산하지 말고, 기억하지 말라는것. 시험 볼때는 오답 노트 작성안해! 이 데이터는 학습에 안쓰려고
        for xb, yb in loader: #with function은 matlab f.read 와 f.close역할. 자동으로 닫아주는 역할임
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            total_loss += loss.item() * xb.size(0) #loss는 데이터, history, CPU 정보 등이 저장되어있음 item으로 데이터만 뽑기
            count += xb.size(0) #loss는 평균값이기 때문에 xb.size(0): 갯수를 곱해서 전체 loss를 호출 따라서 배치가 512인 것과 나머지 488도 똑같이 만들게 하기 위해서
    return total_loss / count #다시 count로 나눠서 loss 평균값구함


best_val = np.inf # 무한대 '가장 낮은 오차'를 찾기 위한 시작점. 작아져야 하니까 업뎃하면 최고기록
best_state = None # 낮은 오차를 가질떄의 가중치를 담아둘 상자

epochs = 50 # 50개 번 한다는거지 머~

for epoch in range(1, epochs + 1):
    model.train() #공부모드 on! 일부 뉴런을 랜덤하게 0으로 만들어 버림. 특정 뉴런에만 의존하지 못하게
    total_loss = 0.0
    count = 0

    for xb, yb in train_loader: #train data Nsim*num_case*0.7/512 만큼 돌아가는듯
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad() #이전 배치에서 계산했던 기울기가 남아있으면 방해되어서 0으로 비움,
        #파이토치는 w.grad = w.grad + new_grad로 동작 메모리 부족시 가상 배치 효과를 내기 위해서
        #즉 배치 사이즈가 1000갠데 메모리가 100개 밖에 안들어가면 zero_grad안하고 배치가 끝나고 100개씩 10번다하고 zero_grad하느늗듯 
        pred = model(xb) 
        loss = criterion(pred, yb) 
        loss.backward() #dLoss/dw_k = dLoss/dpred * dpred/dlayer_n *... * 미분곱
        #끝나면 객체 내부의 모든 파라미터(weight, bias) 등에 대해서 .grad 변수에 계산된 기울기 숫자가 저장됨.
        optimizer.step() #가중치 업데이트

        total_loss += loss.item() * xb.size(0) 
        count += xb.size(0)

    train_loss = total_loss / count 
    val_loss = evaluate(val_loader) #model 검사하고

    if val_loss < best_val:
        best_val = val_loss #좋은 모델로 업데이트 하는것 같음.
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()} 
        #k(key), v(value) : key는 layer1.weight, layer1. bias같은 부품이름
        # Dictionary Comprehension 문법이래 {}: Dictionary type
        # {k:} 딕셔너리의 이름표 레이어의 이름이 들어감
        # v: 이름표에 붙은 실제 데이터 k이름으로 v를 저장해주는 느낌. cpu로 옮긴 후 clone 복사
        # clone을 안하면 v의 주소가 복사되는 느낌이라 model_state_dict()과 같이 변하게 된다.
    print(f"Epoch {epoch:03d} | train {train_loss:.6f} | val {val_loss:.6f}")

model.load_state_dict(best_state)


# -----------------------------
# 7. Test evaluation
# -----------------------------
model.eval()
preds = []
with torch.no_grad(): #model eval과 torch.no_grad는 붙어다님. 데이터 절약 을 위해서..
    for xb, _ in test_loader:
        xb = xb.to(device)
        pred = model(xb).cpu().numpy() #그래프로 그리기 위해서 cpu로 옮기고 numpy 로 변환 배치사이즈와 같은 (512,1)
        preds.append(pred) #배치별로 나온 결과들을 일단 리스트에 차곡차곡 쌓음 [(512,1)배열, (512,1)배열]..

Y_pred_n = np.vstack(preds) #배치 데이터들을 세로로 길게 쌓아 큰 행렬로 만듦 [A;B;C]와 같음 세로병합
Y_pred = Y_pred_n * y_std + y_mean #모델은 0~1사이로 표준화된 숫자만 내뱉음. 표준변차를 곱하고 평균을 더해서 제어 복원하기..
# Y_pred 는 테스트 데이터 갯수 * 출력 변수의 갯수(10) 으로 구성
rmse_each = np.sqrt(np.mean((Y_pred - Y_test) ** 2, axis=0)) #결과 출력 1 오차, 출력 2 오차, ... 출력 10 오차
rmse_total = np.sqrt(np.mean((Y_pred - Y_test) ** 2)) #전체오차

print("Test RMSE each du:") #그래서 du 인 du_k,...du_k+Nc-1까지의 각각의 RMSE를 말하는 거구나
print(rmse_each)
print("Test RMSE total:", rmse_total)


# -----------------------------
# 8. Save model + normalization
# -----------------------------
torch.save(model.state_dict(), "mlp_warmstart_model.pt")

np.savez(
    "mlp_warmstart_norm.npz",
    x_mean=x_mean, #train에 쓴 x_mean과 x_std를 이용해야 학습 때와 똑같은 기준으로 표준화해서 모델로 넘겨줄 수 있음
    x_std=x_std,
    y_mean=y_mean,
    y_std=y_std, #마찬가지로 제어기가 뱉은 인풋값도 정규화해야하니까 불러와줘야 함
)

print("Saved model: mlp_warmstart_model.pt") #pt는 파이토치 전용 확장자. 
print("Saved norm : mlp_warmstart_norm.npz") #npz는 여러개의 넘파이 배열을 하나로 압축해서 저장