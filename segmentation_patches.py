# python main.py --cuda --model fcn8 train --datadir ./MAdata_patches --num-epochs 30 --num-workers 4 --batch-size 1 --steps-plot 50 --steps-save 100

# python main.py --model fcn8 --state segnet2-30-0 eval foo.jpg foo.png


import numpy as np
import torch

import math

from PIL import Image
from argparse import ArgumentParser

from torch.optim import SGD, Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader
# from torchvision.transforms import Compose, CenterCrop, Normalize
from torchvision.transforms import ToTensor, ToPILImage

from piwise.dataset import VOC12
from piwise.network import FCN8, FCN16, FCN32, UNet, PSPNet, SegNet
from piwise.criterion import CrossEntropyLoss2d
from piwise.transform import Colorize
from piwise.visualize import Dashboard


# import MA

from torchvision.transforms import Compose, CenterCrop, Scale, Normalize

from MA.transform import ToLabel, Relabel
# from MA.dataset import MA, eval_ds

from basic_net.dataset import dt_ma

# torch.cuda.set_device(1)

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

eval_input_transform = Compose([
    Scale(256),
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

    # loader = DataLoader(MA('/Users/zhangweidong03/Code/dl/pytorch/github/piwise/MAdata', input_transform, target_transform),
    #                     num_workers=1, batch_size=1, shuffle=True)


    loader = DataLoader(dt_ma(args.root, input_transform, target_transform),
        num_workers=args.workers, batch_size=args.batch, shuffle=True)

    weight = torch.ones(2)
    weight[0] = 0

    use_cuda = False
    if args.cuda:
        criterion = CrossEntropyLoss2d(weight.cuda())
    else:
        criterion = CrossEntropyLoss2d(weight)

    criterion = CrossEntropyLoss2d()

    # optimizer = SGD(model.parameters(), 1e-4, .9, 2e-5)

    optimizer = Adam(model.parameters())
    if args.model.startswith('FCN'):
        optimizer = SGD(model.parameters(), 1e-4, .9, 2e-5)
    if args.model.startswith('PSP'):
        optimizer = SGD(model.parameters(), 1e-2, .9, 1e-4)
    if args.model.startswith('Seg'):
        optimizer = SGD(model.parameters(), 1e-3, .9)

    # for epoch in range(1, 51):
    #     epoch_loss = []
    #
    #     for step, (images, labels) in enumerate(loader):
    #         if use_cuda:
    #             images = images.cuda()
    #             labels = labels.cuda()
    #
    #         inputs = Variable(images)
    #         targets = Variable(labels)
    #         outputs = model(inputs)
    #
    #         optimizer.zero_grad()
    #         loss = criterion(outputs, targets[:, 0])
    #         loss.backward()
    #         optimizer.step()
    #
    #         epoch_loss.append(loss.data[0])
    #
    #         average = sum(epoch_loss) / len(epoch_loss)
    #         print(f'loss: {average} (epoch: {epoch}, step: {step})')


    if args.steps_plot > 0:
        board = Dashboard(args.port)

    for epoch in range(1, args.num_epochs+1):
        epoch_loss = []

        for step, (images, labels) in enumerate(loader):
            if args.cuda:
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
            if args.steps_plot > 0 and step % args.steps_plot == 0:
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
            if args.steps_loss > 0 and step % args.steps_loss == 0:
                average = sum(epoch_loss) / len(epoch_loss)
                print(f'loss: {average} (epoch: {epoch}, step: {step})')
            if args.steps_save > 0 and step % args.steps_save == 0:
                filename = f'{args.model}-{args.exp}-{epoch:03}-{step:04}.pth'
                torch.save(model.state_dict(), filename)
                print(f'save: {filename} (epoch: {epoch}, step: {step})')



# def evaluate(args, model):
#     model.eval()
#
#     im = Image.open(args.image)
#     np_im = np.array(im)
#
#     row = math.ceil(np_im.shape[0]/256)
#     column = math.ceil(np_im.shape[1]/256)
#
#     label = np.zeros(im.shape)
#
#     for i in range(row):
#         for j in range(column):
#             im_patch = np_im[
#                 i*256:(i+1)*256, j*256:(j+1)*256
#             ]
#             im_patch = input_transform(Image.fromarray(im_patch))
#
#             label_patch = model(Variable(im_patch, volatile=True).unsqueeze(0))
#
#             # label_patch = label_patch[0].cpu().max(0)[1].data
#
#             label_patch = color_transform(label_patch[0].data.max(0)[1])
#             label_patch = np.array(image_transform(label_patch))
#             label[i*256:(i+1)*256, j*256:(j+1)*256] = label_patch
#
#     # image = input_transform(Image.open(args.image))
#     # label = model(Variable(image, volatile=True).unsqueeze(0))
#     # label = color_transform(label[0].data.max(0)[1])
#
#     Image.fromarray(label).save(args.label)


# def evaluate(args, model):
#     model.eval()
#
#     loader = DataLoader(
#         eval_ds(args.image, eval_input_transform),
#         num_workers=1, batch_size=1, shuffle=False)
#
#     raw = Image.open(args.image)
#
#     np_img = np.zeros((1024,1024,3), dtype=np.uint8)
#
#     for i, (batch, row, col) in enumerate(loader):
#         label = model(Variable(batch,  volatile=True))
#         label = color_transform(label[0].data.max(0)[1])
#         # image_transform(label).save('{}_{}'.format(i,args.label))
#         img_patch = np.asarray(image_transform(label))
#         np_img[row*256:(row+1)*256, col*256:(col+1)*256] = img_patch
#
#     out_img = Image.fromarray(np_img)
#     out_img = Image.blend(raw, out_img, 0.7)
#     out_img.save('blend_{}'.format(args.label))


def main(args):
    Net = None
    if args.model == 'fcn8':
        Net = FCN8
    if args.model == 'fcn16':
        Net = FCN16
    if args.model == 'fcn32':
        Net = FCN32
    if args.model == 'fcn32':
        Net = FCN32
    if args.model == 'unet':
        Net = UNet
    if args.model == 'pspnet':
        Net = PSPNet
    if args.model == 'segnet':
        Net = SegNet
    assert Net is not None, f'model {args.model} not available'

    model = Net(NUM_CLASSES)

    if args.cuda:
        print('use cuda')
        model = model.cuda()

    model = torch.nn.DataParallel(model).cuda()

    if args.state:
        try:
            model.load_state_dict(torch.load(args.state))
        except AssertionError:
            model.load_state_dict(torch.load(args.state,
                map_location=lambda storage, loc: storage))

    if args.mode == 'eval':
        print('eval is not ready!')
        # evaluate(args, model)
    if args.mode == 'train':
        train(args, model)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--model', required=True)
    parser.add_argument('--state')

    subparsers = parser.add_subparsers(dest='mode')
    subparsers.required = True

    parser_eval = subparsers.add_parser('eval')
    parser_eval.add_argument('image')
    parser_eval.add_argument('label')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--port', type=int, default=80)
    parser_train.add_argument('--root', required=True)
    parser_train.add_argument('--epochs', type=int, default=32)
    parser_train.add_argument('--workers', type=int, default=4)
    parser_train.add_argument('--batch', type=int, default=1)
    parser_train.add_argument('--steps-loss', type=int, default=50)
    parser_train.add_argument('--steps-plot', type=int, default=0)
    parser_train.add_argument('--steps-save', type=int, default=500)
    parser_train.add_argument('--exp', default='segmentation_patches')

    main(parser.parse_args())