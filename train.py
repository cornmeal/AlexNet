import os

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
# 数据集划分
from torch.utils.data import random_split
from torchvision import datasets
# 图像预处理
from torchvision import transforms
# 绘图工具库
import matplotlib.pyplot as plt
from model import AlexNet


# 训练函数
def train(dataLoader, model, loss_fn, optimizer):
    # 训练模型时启用BatchNormalization和Dropout, 将BatchNormalization和Dropout置为True
    model.train()
    total = 0  # 已训练的样本总数
    total_loss = 0.0
    step = 0
    correct = 0  # 预测正确的样本数
    total_f1 = 0.0  # F1-score汇总
    for batch, (inputs, labels) in enumerate(dataLoader):
        # 把模型部署到device上
        inputs, labels = inputs.to(device), labels.to(device)
        # 初始化梯度
        optimizer.zero_grad()
        # 保存训练结果
        outputs = model(inputs)
        # 计算损失和
        loss = loss_fn(outputs, labels)
        # 获取最大概率的预测结果
        # dim=1表示返回每一行的最大值对应的列下标
        predict = outputs.argmax(dim=1)
        # 统计样本总数
        total += labels.size(0)
        # 计算预测正确的样本个数
        correct += (predict == labels).sum().item()
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()

        step += 1
        total_loss += loss.item()
        total_f1 += f1_score(labels, predict, average='macro')

        # 每10轮展示参数
        if batch % 10 == 0:
            # loss.item()表示当前loss的数值
            print("Train Data:   Loss: {:.6f}, accuracy: {:.6f}%,  F1-score: {:.6f}".format(loss.item(), 100 * (correct / total), total_f1 / step))
    return total_loss / total, correct / total, total_f1 / step


# 验证函数
def val(model, loss_fn, dataLoader):
    # 模型评估模式, 因为调用eval()将不启用 BatchNormalization 和 Dropout, BatchNormalization和Dropout置为False
    model.eval()
    # 统计模型正确率, 设置初始值
    correct = 0.0
    test_loss = 0.0
    total = 0
    total_f1 = 0.0
    step = 0
    # torch.no_grad将不会计算梯度, 也不会进行反向传播
    with torch.no_grad():
        for data, label in dataLoader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            loss = loss_fn(output, label)
            test_loss += loss.item()
            # 0是每列的最大值，1是每行的最大值
            predict = output.argmax(dim=1)
            total += label.size(0)
            # 计算正确数量
            correct += (predict == label).sum().item()

            step += 1
            total_f1 += f1_score(label, predict, average='macro')

        # 计算损失值
        print("Val Data:   Loss: {:.6f}, accuracy: {:.6f}%,  F1-score: {:.6f}".format(test_loss / total, 100 * (correct / total), total_f1 / step))
    return test_loss / total, correct / total, total_f1 / step


def matplot_comparison(train_data, val_data, xlabel, ylabel, title):
    plt.plot(train_data, label='train')
    plt.plot(val_data, label='val')
    plt.legend(loc='best')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    # compose串联多个transform操作
    train_transform = transforms.Compose([
        # 随机旋转
        transforms.RandomHorizontalFlip(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # 加载数据集和验证集
    train_data = datasets.ImageFolder('./myData/train', transform=train_transform)
    val_data = datasets.ImageFolder('./myData/val', transform=train_transform)

    # train_data, val_data = random_split(dataset, (4396, 1099), generator=torch.Generator().manual_seed(42))

    # 加载数据集
    train_dataLoader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    val_dataLoader = torch.utils.data.DataLoader(val_data, batch_size=32, shuffle=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 创建模型部署到device上
    model = AlexNet().to(device)
    # 交叉熵损失函数
    loss_func = nn.CrossEntropyLoss()
    # 定义优化器
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    # 学习率每隔step_size就变为原来的gamma倍
    # lr_scheduler = lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)

    loss_train = []
    acc_train = []
    avg_f1_train = []
    loss_val = []
    acc_val = []
    avg_f1_val = []

    epochs = 2
    for epoch in range(epochs):
        print(f'Epoch:[{epoch + 1}/{epochs}]\n------------------------------------------')
        train_loss, train_acc, avg_f1 = train(train_dataLoader, model, loss_func, opt)
        val_loss, val_acc, val_f1 = val(model, loss_func, val_dataLoader)
        # lr_scheduler.step()

        loss_train.append(train_loss)
        acc_train.append(train_acc)
        avg_f1_train.append(avg_f1)

        loss_val.append(val_loss)
        acc_val.append(val_acc)
        avg_f1_val.append(val_f1)

    # 绘图
    matplot_comparison(loss_train, loss_val, 'epoch', 'loss', 'Loss Comparison')
    matplot_comparison(acc_train, acc_val, 'epoch', 'accuracy', 'Accuracy Comparison')
    matplot_comparison(avg_f1_train, avg_f1_val, 'epoch', 'F1-score', 'F1-score Comparison')

    # 保存模型
    if not os.path.exists('models'):
        os.mkdir('models')
    torch.save(model.state_dict(), 'models/AlexModel.pth')

