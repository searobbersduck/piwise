import argparse

import torch

from torch.optim import SGD

from torchvision.transforms import Compose, CenterCrop, Scale, Normalize, ToTensor, ToPILImage
from MA.transform import ToLabel, Relabel

from basic_net.dataset import dt_ma
from torch.utils.data import DataLoader
from torch.autograd import Variable

from piwise.transform import Colorize
from piwise.criterion import CrossEntropyLoss2d
from piwise.visualize import Dashboard

from basic_net.network import FCN8

import torch.backends.cudnn as cudnn


def arg_parse():
    parser = argparse.ArgumentParser(description='Segmentation algorithm parameters')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--state')

    subparsers = parser.add_subparsers(dest='mode')
    subparsers.required = True

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--port', type=int, default=80)
    parser_train.add_argument('--root', required=True)
    parser_train.add_argument('--epochs', type=int, default=100)
    parser_train.add_argument('--workers', type=int, default=4)
    parser_train.add_argument('--batch', type=int, default=1)
    parser_train.add_argument('--steps-loss', type=int, default=50)
    parser_train.add_argument('--steps-plot', type=int, default=0)
    parser_train.add_argument('--steps-save', type=int, default=500)
    parser_train.add_argument('--exp', default='default')

    return parser.parse_args()

NUM_CHANNELS = 3
NUM_CLASSES = 2

color_transform = Colorize()
image_transform = ToPILImage()

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
    Relabel(255, 1)
])


def get_model():
    Net = FCN8
    model = Net(NUM_CLASSES, './vgg_16.pth')
    return model


def train(opt, model, use_cuda):
    loader = DataLoader(dt_ma(opt.root, input_transform, target_transform),
                        num_workers=opt.workers,
                        batch_size=opt.batch,
                        pin_memory=True,
                        shuffle=True)
    weight = torch.ones(2)
    weight[0] = 0

    if use_cuda:
        criterion = CrossEntropyLoss2d(weight.cuda())
    else:
        criterion = CrossEntropyLoss2d(weight)

    criterion = CrossEntropyLoss2d().cuda()

    optimizer = SGD(model.parameters(), 1e-4, .9, 2e-5)

    if opt.steps_plot > 0:
        board = Dashboard(opt.port)

    for epoch in range(opt.epochs+1):
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

            if opt.steps_plot > 0 and step % opt.steps_plot == 0:
                image = inputs[0].cpu().data
                image[0] = image[0] * .229 + .485
                image[1] = image[1] * .224 + .456
                image[2] = image[2] * .225 + .406
                board.image(image,
                            f'input (epoch: {epoch}, step: {step})')
                board.image(color_transform(outputs[0].cpu().max(0)[1].data),
                            f'output (epoch: {epoch}, step: {step})')
                board.image(color_transform(targets[0].cpu().data),
                            f'target (epoch: {epoch}, step: {step})')
            if opt.steps_loss > 0 and step % opt.steps_loss == 0:
                average = sum(epoch_loss) / len(epoch_loss)
                print(f'loss: {average} (epoch: {epoch}, step: {step})')
            if opt.steps_save > 0 and step % opt.steps_save == 0:
                filename = f'fcn8-{opt.exp}-{epoch:03}-{step:04}.pth'
                torch.save(model.state_dict(), filename)
                print(f'save: {filename} (epoch: {epoch}, step: {step})')


def main():
    print('===> Parsing options:')
    opt = arg_parse()
    print(opt)
    cudnn.benchmark = True
    torch.manual_seed(1)

    use_cuda = opt.cuda and torch.cuda.is_available()

    if use_cuda:
        print('use cuda')
    else:
        print('cuda is not used')

    model = get_model()

    # if opt.state:
    #     model.load_state_dict(torch.load(opt.state))

    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()

    if opt.state:
        model.load_state_dict(torch.load(opt.state))

    if opt.mode == 'train':
        train(opt, model, use_cuda)

main()