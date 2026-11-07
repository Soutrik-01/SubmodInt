import torch
from PIL import Image
import open_clip
import seaborn as sns 
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
import torchvision 
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
import torch.nn.functional as F

templates80 = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',]

device = torch.device('cuda:0')
model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('ViT-bigG-14', pretrained='laion2b_s39b_b160k')
tokenizer = open_clip.get_tokenizer('ViT-bigG-14')

def get_color_distortion(s=1.0):
	color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
	rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
	rnd_gray = transforms.RandomGrayscale(p=0.2)
	color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
	return color_distort

def get_loader(jitter_strength=0.1, bsz=1024, strong=True):
    if strong:
        color_transform = [get_color_distortion(jitter_strength)]
        additional_transforms = transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomRotation(5),
                        transforms.Compose(color_transform),
                    ])
    else:
        additional_transforms = transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomRotation(5),
                    ])
        
    final_preprocess = transforms.Compose(preprocess_val.transforms[:-1] + additional_transforms.transforms + [preprocess_val.transforms[-1]])
    dataset_train = torchvision.datasets.CIFAR100(root = '../data', download = True, train = True, transform = final_preprocess)
    dataset_test = torchvision.datasets.CIFAR100(root = '../data', download = True, train = False, transform = preprocess_val)
    
    trainloader = DataLoader(dataset_train, batch_size=bsz, shuffle=False, num_workers=12)
    testloader = DataLoader(dataset_test, batch_size=bsz, shuffle=False, num_workers=12)
    
    return trainloader, testloader

@torch.no_grad()
def get_H_Y(model, loader):
    H_train = []
    Y_train = []
    model.eval()
    with torch.no_grad():
        for _, (x, y) in tqdm(enumerate(loader)):
            x = x.to(device, non_blocking=True); Y_train.append(y)
            Y_train.append(y)
            H_train.append(model.encode_image(x).cpu())
    torch.cuda.empty_cache()
    H_train = torch.cat(H_train, 0)
    Y_train = torch.cat(Y_train, 0)
    return H_train.cpu().numpy(), Y_train.numpy()
    

model = model.to(device); model = model.eval()


_, testloader = get_loader()
X_test, Y_test = get_H_Y(model, testloader)
np.save("X_test.npy", X_test)
np.save("Y_test.npy", Y_test)



trainloader_weak, _ = get_loader(strong=False)
trainloader_strong, _ = get_loader(jitter_strength=0.1)

H_all = []

for i, seed in enumerate([10, 20, 30, 40, 50]):
    print("Seed: ", seed)
    if seed == 10:
        loader = trainloader_weak 
    else:
        loader = trainloader_strong 
    
    H_, Y_train = get_H_Y(model, loader)
    H_all.append(H_)

np.save("Y_train.npy", Y_train)
X_train = np.stack(H_all, axis=-1)
np.save("X_train.npy", X_train)
