from MA.dataset import eval_ds

from torch.utils.data import DataLoader

from torchvision.transforms import Compose, CenterCrop, Scale, Normalize, ToTensor

from torch.autograd import Variable

input_transform = Compose([
    Scale(256),
    ToTensor(),
    Normalize([.485, .456, .406], [.229, .224, .225]),
])

imagepath = '/Users/zhangweidong03/Code/dl/pytorch/github/pi/piwise/MAdata/images/C0000887.jpg'

loader = DataLoader(eval_ds('/Users/zhangweidong03/Code/dl/pytorch/github/pi/piwise/MAdata/images/C0000887.jpg', input_transform),
                    num_workers=1, batch_size=10, shuffle=False)


for i, image in enumerate(loader):
    var = Variable(image)
    print(i)