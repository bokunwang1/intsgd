import numpy as np
import torch
from utils import accuracy_and_loss


def run_sgd(net, device=None, trainloader=None, testloader=None, N_train=5, batch_size=10, n_epoch=5, epoch_lrs=[],
            optimizer=None, checkpoint=125, noisy_train_stat=True):
    losses = []
    train_acc = []
    test_losses = []
    test_acc = []
    it_train = []
    it_test = []
    grad_norms = []
    net.train()  # Enable things such as dropout
    criterion = torch.nn.CrossEntropyLoss()

    lr_idx = 0
    for epoch in range(n_epoch):  # loop over the dataset multiple times
        if len(epoch_lrs) > lr_idx:
            if epoch == epoch_lrs[lr_idx][0]:
                for group in optimizer.param_groups:
                    group['lr'] = epoch_lrs[lr_idx][1]
                lr_idx += 1

        running_loss = 0.0

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if noisy_train_stat and (i % 10) == 0:
                losses.append(loss.cpu().item())
                it_train.append(epoch + i * batch_size / N_train)

            # print('i: {0}, ckpt: {1}'.format(i, checkpoint))

            if i % checkpoint == (checkpoint - 1):
                if running_loss / checkpoint < 0.01:
                    print('[%d, %5d] loss: %.4f' %
                          (epoch + 1, i + 1, running_loss / checkpoint))
                else:
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / checkpoint))
                running_loss = 0.0
                test_a, test_l = accuracy_and_loss(net, testloader, device, criterion)
                print('[%d, %5d] Test Accuracy: %.3f' %
                      (epoch + 1, i + 1, test_a))
                test_acc.append(test_a)
                test_losses.append(test_l)
                grad_norms.append(np.sum([p.grad.data.norm().item() for p in net.parameters()]))
                net.train()
                it_test.append(epoch + i * batch_size / N_train)

        if not noisy_train_stat:
            it_train.append(epoch)
            train_a, train_l = accuracy_and_loss(net, trainloader, device, criterion)
            train_acc.append(train_a)
            losses.append(train_l)
            net.train()

    print('Finished Training')
    return (np.array(losses), np.array(test_losses), np.array(train_acc), np.array(test_acc),
            np.array(it_train), np.array(it_test), np.array(grad_norms))
