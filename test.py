import torch
from sklearn.metrics import f1_score
from model import AlexNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
            total_f1 += f1_score(label.cpu(), predict.cpu(), average='macro')

        # 计算损失值
        print("Val Data:   Loss: {:.6f}, accuracy: {:.6f}%,  F1-score: {:.6f}".format(test_loss / step, 100 * (correct / total), total_f1 / step))
    return test_loss / step, correct / total, total_f1 / step


if __name__ == '__main__':
    model = AlexNet().to(device)
    # 加载模型
    model.load_state_dict(torch.load('./models/AlexModel.pth'))
