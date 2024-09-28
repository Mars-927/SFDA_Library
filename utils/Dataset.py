import numpy as np
import torch
import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from torchvision.transforms import v2

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
    
def make_dataset(image_list, labels=None):
    if labels:
        len_ = len(image_list)
        images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
        if len(image_list[0].split()) > 2:
            images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list] 
        else:
            images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images

def RGB_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class dataset_read(Dataset):
    def __init__(self, image_list, image_base, is_preread = True, only_clear = True):
        self.imgs = make_dataset(image_list)
        self.is_preread = is_preread
        self.define_transform()

        if is_preread:
            self.pre_read = []
            images_size = len(self.imgs)
            for index in tqdm(range(images_size),desc=f"read dataset"):
                item = self.imgs[index]
                imgs_path, imgs_label = item
                imgs = RGB_loader(os.path.join(image_base,imgs_path))
                if only_clear:
                    self.pre_read.append({
                        "img":self.clear_transform(imgs),
                        "label":imgs_label,
                        "path":imgs_path,
                        "index":index
                    })
                else:
                    self.pre_read.append({
                        "img":self.clear_transform(imgs),
                        "weak_img":self.weak_transform(imgs),
                        "strong_img":self.strong_transform(imgs),
                        "label":imgs_label,
                        "path":imgs_path,
                        "index":index
                    })

        else:
            self.image_base = image_base

    def __getitem__(self, index):
        if self.is_preread:
            return self.pre_read[index]
        else:
            imgs_path, imgs_label = self.imgs[index]
            imgs = RGB_loader(os.path.join(self.image_base,imgs_path))
            return {
                    "img":self.transform(imgs),
                    "label":imgs_label,
                    "path":imgs_path,
                    "index":index
                }
    
    def __len__(self):
        return len(self.imgs)

    def define_transform(self):
        self.strong_transform = transforms.Compose([
            transforms.Resize([224,224]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            v2.RandAugment(),
            transforms.ToTensor(), 
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
        ])
        self.weak_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), 
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
            ])
        self.clear_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(), 
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        ])

def get_dataloader(args, domain, is_preread = True):
    txt_path = os.path.join(args.dataset_path,f"{domain}.txt")
    image_path = os.path.join(args.image_root,domain)
    dataset_dirt = {}
    dataset_txt = open(txt_path).readlines()
    dataset_size = len(dataset_txt)
    train_size = int(dataset_size * 0.8)
    test_size = dataset_size - train_size
    train_list, test_list = torch.utils.data.random_split(dataset_txt, [train_size, test_size])

    train_dataset = dataset_read(train_list, image_base=image_path, is_preread=is_preread,only_clear=False)
    test_dataset = dataset_read(test_list, image_base=image_path, is_preread=is_preread)
    all_dataset = dataset_read(dataset_txt, image_base=image_path, is_preread=is_preread)
    dataset_dirt["train"] = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, pin_memory=True)
    dataset_dirt["test"] = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, pin_memory=True)
    dataset_dirt["all"] = DataLoader(all_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, pin_memory=True)
    args.train_dataset_size = len(train_dataset)
    args.test_dataset_size = len(test_dataset)
    args.all_dataset_size = len(train_dataset)
    return dataset_dirt

class dataset_read_CTRR(Dataset):
    def __init__(self, image_list, image_base, is_preread = True, only_clear = True):
        self.pre_read = []
        self.is_preread = is_preread
        self.imgs = make_dataset(image_list)
        self.images_size = len(self.imgs)
        self.only_clear = only_clear
        self.define_transform()

        if is_preread:
            for index in tqdm(range(self.images_size),desc=f"read dataset"):
                item = self.imgs[index]
                imgs_path, imgs_label = item
                imgs = RGB_loader(os.path.join(image_base,imgs_path))
                if only_clear:
                    self.pre_read.append({
                        "img":self.clear_transform(imgs),
                        "label":imgs_label,
                        "path":imgs_path,
                        "index":index
                    })
                else:
                    self.pre_read.append({
                        "img":self.clear_transform(imgs),
                        "train_cls_transformcon":self.train_cls_transformcon(imgs),
                        "train_transforms_1":self.train_transforms(imgs),
                        "train_transforms_2":self.train_transforms(imgs),
                        "test_transform":self.test_transform(imgs),
                        "label":imgs_label,
                        "path":imgs_path,
                        "index":index
                    })
        else:
            # only RGB_loader
            for index in tqdm(range(self.images_size),desc=f"read dataset"):
                item = self.imgs[index]
                imgs_path, _ = item
                imgs = RGB_loader(os.path.join(image_base,imgs_path))
                self.pre_read.append(imgs)

    def __getitem__(self, index):
        if self.is_preread:
            return self.pre_read[index]
        else:
            imgs_path, imgs_label = self.imgs[index]
            imgs = self.pre_read[index]
            if self.only_clear:
                return {
                        "img":self.clear_transform(imgs),
                        "label":imgs_label,
                        "path":imgs_path,
                        "index":index
                    }
            else:
                return {
                        "img":self.clear_transform(imgs),
                        "train_cls_transformcon":self.train_cls_transformcon(imgs),
                        "train_transforms_1":self.train_transforms(imgs),
                        "train_transforms_2":self.train_transforms(imgs),
                        "label":imgs_label,
                        "path":imgs_path,
                        "index":index
                    }

    def __len__(self):
        return len(self.imgs)

    def define_transform(self):
        self.train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        ])
        self.train_cls_transformcon = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
            ])
        self.test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
            ])
        self.clear_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(), 
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        ])

def get_dataloader_CTRR(args, domain, is_preread = True):
    txt_path = os.path.join(args.dataset_path,f"{domain}.txt")
    image_path = os.path.join(args.image_root,domain)
    dataset_dirt = {}
    dataset_txt = open(txt_path).readlines()
    dataset_size = len(dataset_txt)
    train_size = int(dataset_size * 0.8)
    test_size = dataset_size - train_size
    train_list, test_list = torch.utils.data.random_split(dataset_txt, [train_size, test_size])

    train_dataset = dataset_read_CTRR(train_list, image_base=image_path, is_preread=is_preread,only_clear=False)
    test_dataset = dataset_read_CTRR(test_list, image_base=image_path, is_preread=is_preread)
    all_dataset = dataset_read_CTRR(dataset_txt, image_base=image_path, is_preread=is_preread)
    dataset_dirt["train"] = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, pin_memory=True)
    dataset_dirt["test"] = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, pin_memory=True)
    dataset_dirt["all"] = DataLoader(all_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, pin_memory=True)
    args.train_dataset_size = len(train_dataset)
    args.test_dataset_size = len(test_dataset)
    args.all_dataset_size = len(all_dataset)
    return dataset_dirt



class dataset_shot(Dataset):
    def __init__(self, image_list, image_base, isTrian = True):
        self.imgs = make_dataset(image_list)
        self.imgs_raw = []
        self.imgs_test = []
        self.images_size = len(self.imgs)
        self.isTrian = isTrian
        self.get_transform()
        for index in tqdm(range(self.images_size),desc=f"read dataset"):
            item = self.imgs[index]
            imgs_path, _ = item
            imgs = RGB_loader(os.path.join(image_base,imgs_path))
            self.imgs_raw.append(imgs)
            if not isTrian:
                # 预读images test数据
                self.imgs_test.append(self.test_transform(imgs))

    def __getitem__(self, index):
        _, imgs_label = self.imgs[index]
        if self.isTrian:
            img = self.train_transform(self.imgs_raw[index])
        else:
            img = self.imgs_test[index]
        return {
            "img":img,
            "label":imgs_label,
            "index": index,
            }

    def __len__(self):
        return len(self.imgs)

    def get_transform(self):
        self.train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])]
            )
        
        self.test_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])]
            )


class dataset_shotSource(Dataset):
    def __init__(self, image_list, image_base, isTrian = True):
        self.imgs = make_dataset(image_list)
        self.imgs_raw = []
        self.imgs_test = []
        self.images_size = len(self.imgs)
        self.isTrian = isTrian
        self.get_transform()
        for index in tqdm(range(self.images_size),desc=f"read dataset"):
            item = self.imgs[index]
            imgs_path, _ = item
            imgs = RGB_loader(os.path.join(image_base,imgs_path))
            self.imgs_raw.append(imgs)
            if not isTrian:
                # 预读images test数据
                self.imgs_test.append(self.test_transform(imgs))

    def __getitem__(self, index):
        _, imgs_label = self.imgs[index]
        if self.isTrian:
            img = self.train_transform(self.imgs_raw[index])
        else:
            img = self.imgs_test[index]
        return {
            "img":img,
            "label":imgs_label,
            }

    def __len__(self):
        return len(self.imgs)

    def get_transform(self):
        self.train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])]
            )
        
        self.test_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])]
            )


class dataset_guidingPseudoSFDA(Dataset):
    def __init__(self, image_list, image_base,isTrian = True):
        self.imgs = make_dataset(image_list)
        self.imgs_raw = []
        self.imgs_test = []
        self.images_size = len(self.imgs)
        self.isTrian = isTrian
        self.get_transform()
        for index in tqdm(range(self.images_size),desc=f"read dataset"):
            item = self.imgs[index]
            imgs_path, _ = item
            imgs = RGB_loader(os.path.join(image_base,imgs_path))
            self.imgs_raw.append(imgs)
        if not isTrian:
            self.imgs_test.append(self.test_transform(imgs))

    def __getitem__(self, index):
        _, imgs_label = self.imgs[index]
        if self.isTrian:
            imgs = self.imgs_test[index]
            weak_augmented = self.train_weak_transform(imgs)
            strong_augmented = self.strong_transform(imgs)
            strong_augmented2 = self.strong_transform(imgs)
            return weak_augmented, strong_augmented, imgs_label, index, None, strong_augmented2
        else:
            weak_augmented = self.imgs_test[index]
            return weak_augmented, None, imgs_label, index, None, None
        
        
    def __len__(self):
        return len(self.imgs)
    
    def get_transform(self):
        self.test_transform = transforms.Compose([
            transforms.Resize(256), 
            transforms.CenterCrop(224), 
            transforms.ToTensor(), 
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])]
            )
        self.train_weak_transform = transforms.Compose([
            transforms.Resize(256), 
            transforms.RandomCrop(224), 
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor(), 
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])]
            )
        self.strong_transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],p=0.8,),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]
            )
                  

def get_dataloader_select(args, domain):
    txt_path = os.path.join(args.dataset_path, f"{domain}.txt")
    image_path = os.path.join(args.image_root, domain)
    dataset_dirt = {}
    dataset_txt = open(txt_path).readlines()
    dataset_size = len(dataset_txt)


    if args.method == 'guidingPseudoSFDA':
        dataset_read = dataset_guidingPseudoSFDA
        

    elif args.method == 'source_shot':
        # for shot source, train source is 9:1 split train and test
        # for shot source, test target in only source use all dataset
        dataset_read = dataset_shotSource
        train_size = int(dataset_size * 0.9)
        test_size = dataset_size - train_size
        train_list, test_list = torch.utils.data.random_split(dataset_txt, [train_size, test_size])

    elif args.method == 'shot':
        # for shot domain adaptation, train and test use all dataset, no split!
        dataset_read = dataset_shot
        train_list, test_list = dataset_txt, dataset_txt
        

    train_dataset = dataset_read(train_list, image_path, True)
    test_dataset = dataset_read(test_list, image_path, False)
    all_dataset = dataset_read(dataset_txt, image_path, False)

    dataset_dirt["train"] = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, pin_memory=True)
    dataset_dirt["test"] = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, pin_memory=True)
    dataset_dirt["all"] = DataLoader(all_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, pin_memory=True)

    args.test_dataset_size = len(test_dataset)
    args.train_dataset_size = len(train_dataset)

    return dataset_dirt