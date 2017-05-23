import MA

import torch

from argparse import ArgumentParser

from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.autograd import Variable

from torchvision.transforms import Compose, CenterCrop, Scale, Normalize
from torchvision.transforms import ToTensor, ToPILImage

from MA.transform import ToLabel, Relabel
from MA.dataset import MA

from piwise.network import FCN8
from piwise.criterion import CrossEntropyLoss2d

NUM_CHANNELS = 3
NUM_CLASSES = 3

input_transform = Compose([
    Scale(256),
    CenterCrop(256),
    ToTensor(),
    Normalize([.485, .456, .406], [.229, .224, .225]),
])

target_transform = Compose([
    Scale(256),
    CenterCrop(256),
    ToLabel(),
    Relabel(255, 1),
])


def train(args, model):
    model.train()

    loader = DataLoader(MA('/Users/zhangweidong03/Code/dl/pytorch/github/piwise/MAdata', input_transform, target_transform),
                        num_workers=1, batch_size=1, shuffle=True)


    weight = torch.ones(2)
    weight[0] = 0
    use_cuda = False
    if use_cuda:
        criterion = CrossEntropyLoss2d(weight.cuda())
    else:
        criterion = CrossEntropyLoss2d(weight)

    criterion = CrossEntropyLoss2d()

    optimizer = SGD(model.parameters(), 1e-4, .9, 2e-5)

    for epoch in range(1, 51):
        epoch_loss = []

        for step, (images, labels) in enumerate(loader):
            if use_cuda:
                images = images.cuda()
                labels = labels.cuda()

            inputs = Variable(images)
            targets = Variable(labels)
            outputs = model(inputs)

            optimizer.zero_grad()
            loss = criterion(outputs, targets[:, 0])
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.data[0])

            average = sum(epoch_loss) / len(epoch_loss)
            print(f'loss: {average} (epoch: {epoch}, step: {step})')


def main(args):
    Net = FCN8

    model = Net(NUM_CLASSES)

    # if args.cuda:
    #     model = model.cuda()


    train(args, model)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    main(parser.parse_args())

