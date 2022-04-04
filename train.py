import os

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
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
def train(dataLoader, model, loss_fn, optimizer, epoch):
    # 训练模型时启用BatchNormalization和Dropout, 将BatchNormalization和Dropout置为True
    model.train()
    total = 0
    step = 0
    loss, accuracy, total_f1 = 0.0, 0.0, 0.0
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
        accuracy += (predict == labels).sum().item()
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()

        f1 = f1_score(labels, predict, average='macro')
        total_f1 += f1

        step += 1
        # 每10轮计算一次准确率
        if batch % 10 == 0:
            # loss.item()表示当前loss的数值
            print("Train Epoch:batch[{}:{}] \t Loss: {:.6f}, accuracy: {:.6f}%".format(epoch, batch + 1, loss.item(), 100 * (accuracy / total)))
            print(f'Train Epoch:batch[{epoch}:{batch + 1}]  f1_score:{total_f1 / step}')
            loss_train.append(loss.item())
            acc_train.append(accuracy / total)
    return loss.item(), accuracy / total, total_f1 / step


# 验证函数
def val(model, loss_fn, dataLoader):
    # 模型评估模式, 因为调用eval()将不启用 BatchNormalization 和 Dropout, BatchNormalization和Dropout置为False
    model.eval()
    # 统计模型正确率, 设置初始值
    correct = 0.0
    test_loss = 0.0
    total = 0
    # torch.no_grad将不会计算梯度, 也不会进行反向传播
    with torch.no_grad():
        for data, label in dataLoader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            test_loss += loss_fn(output, label).item()
            # 0是每列的最大值，1是每行的最大值
            predict = output.argmax(dim=1)
            total += label.size(0)
            # 计算正确数量
            correct += (predict == label).sum().item()

        # 计算损失值
        print("test_average_loss: {:.6f}, accuracy: {:.6f}%".format(test_loss/total, 100*(correct/total)))
    return test_loss / total, correct / total


if __name__ == '__main__':
    # compose串联多个transform操作
    train_transform = transforms.Compose([
        # 随机旋转
        transforms.RandomHorizontalFlip(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # 划分数据集和验证集
    train_data = datasets.ImageFolder('./myData/train', transform=train_transform)
    # val_data = datasets.ImageFolder('./myData/val', transform=train_transform)

    # train_data, val_data = random_split(dataset, (4396, 1099), generator=torch.Generator().manual_seed(42))

    # 加载数据集
    train_dataLoader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    # val_dataLoader = torch.utils.data.DataLoader(val_data, batch_size=32, shuffle=True)

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
    avg_f1_list = []
    loss_val = []
    acc_val = []

    epoch = 1
    for t in range(epoch):
        print(f'epoch{t + 1}\n---------')
        train_loss, train_acc, avg_f1 = train(train_dataLoader, model, loss_func, opt, t + 1)
        # val_loss, val_acc = val(model, loss_func, val_dataLoader)
        # lr_scheduler.step()

        loss_train.append(train_loss)
        acc_train.append(train_acc)
        avg_f1_list.append(avg_f1)
        # loss_val.append(val_loss)
        # acc_val.append(val_acc)

    # 保存模型
    if not os.path.exists('models'):
        os.mkdir('models')
    torch.save(model.state_dict(), 'models/AlexModel.pth')

    plt.subplot(2, 1, 1)
    plt.plot(loss_train)
    plt.title('Loss')
    plt.show()

    plt.subplot(2, 1, 2)
    plt.plot(acc_train)
    plt.title('Accuracy')
    plt.show()

    plt.subplot(2, 1, 2)
    plt.plot(avg_f1_list)
    plt.title('f1_score')
    plt.show()

    # plt.subplot(2, 1, 1)
    # plt.plot(loss_val)
    # plt.title('Loss')
    # plt.show()
    # plt.subplot(2, 1, 2)
    # plt.plot(acc_val)
    # plt.title('Accuracy')
    # plt.show()
