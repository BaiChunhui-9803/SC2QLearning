import torch
import numpy as np


def mtx_similar1(arr1: np.ndarray, arr2: np.ndarray) -> float:
    '''
    计算矩阵相似度的一种方法。
    将矩阵展平成向量，计算向量的乘积除以模长。
    :param arr1:矩阵1
    :param arr2:矩阵2
    :return:实际是夹角的余弦值，ret = (cos+1)/2
    '''
    farr1 = arr1.ravel()
    farr2 = arr2.ravel()
    len1 = len(farr1)
    len2 = len(farr2)
    if len1 > len2:
        farr1 = farr1[:len2]
    else:
        farr2 = farr2[:len1]
    numer = np.sum(farr1 * farr2)
    denom = np.sqrt(np.sum(farr1 ** 2) * np.sum(farr2 ** 2))
    similar = numer / denom
    return (similar + 1) / 2  # 姑且把余弦函数当线性


def mtx_similar2(arr1: np.ndarray, arr2: np.ndarray) -> float:
    '''
    如果矩阵大小不一样会在左上角对齐，截取二者最小的相交范围。
    :param arr1:矩阵1
    :param arr2:矩阵2
    :return:相似度（0~1之间）
    '''
    if arr1.shape != arr2.shape:
        minx = min(arr1.shape, arr2.shape)
        miny = min(arr1.shape, arr2.shape)
        differ = arr1[:minx, :miny] - arr2[:minx, :miny]
    else:
        differ = arr1 - arr2
    numera = np.sum(differ ** 2)
    denom = np.sum(arr1 ** 2)
    similar = 1 - (numera / denom)
    return similar


# 模型实例化
model = Your_model().to(args.device)

# 设置损失函数和优化器
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(Your_model.parameters(), lr=args.learning_rate)

train_loss = []
valid_loss = []
train_epochs_loss = []
valid_epochs_loss = []

early_stopping = EarlyStopping(patience=args.patience, verbose=True)

# 开始训练以及调整lr
for epoch in range(args.epochs):

    # 训练模型
    model.train()
    train_loss_batch = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(args.device), target.to(args.device)
        optimizer.zero_grad()
        output = model(data)
        loss_train_batch = criterion(output, target)
        loss_train_batch.backward()
        optimizer.step()
        train_loss_batch.append(loss_train_batch.item())
    train_loss.append(np.mean(train_loss_batch))

    # 验证模型
    model.eval()
    valid_loss_batch = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(valid_loader):
            data, target = data.to(args.device), target.to(args.device)
            output = model(data)
            loss_valid_batch = criterion(output, target)
            valid_loss_batch.append(loss_valid_batch.item())
        valid_loss.append(np.mean(valid_loss_batch))

    # 记录每个epoch的loss值，用于画图
    train_epochs_loss.append(train_loss[-1])
    valid_epochs_loss.append(valid_loss[-1])

    # early stopping判断是否停止训练
    early_stopping(valid_loss[-1], model)

    if early_stopping.early_stop:
        print("Early stopping")
        break