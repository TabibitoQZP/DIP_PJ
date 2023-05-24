from transformers import SamModel, SamProcessor
import os
import cv2
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np

import argparse

class ViTSum(nn.Module):
    def __init__(self, samtype="facebook/sam-vit-base", drop=0.2):
        """
        sametype: mainly facebook/sam-vit-base, since we need to fine-tune it.
        drop: keep 1-drop parameters training.
        """
        super().__init__()

        # load sam model
        self.sam = SamModel.from_pretrained(samtype)

        # lock some sam layers. keep following layers train, and lock rest layers.
        trainlayers = ["mask_decoder.transformer", "mask_decoder.upscale_conv1", "mask_decoder.upscale_conv2",
                   "mask_decoder.upscale_layer_norm", "mask_decoder.output_hypernetworks_mlps"]
        for name, param in self.sam.named_parameters():
            param.requires_grad = False
            for l in trainlayers:
                if name.startswith(l):
                    param.requires_grad = True
                    break

        # compress the masks into one layer.
        self.conv = nn.Conv3d(3, 1, 3, padding=(0, 1, 1))

    def forward(self, pv, ib):
        samoutput = self.sam(pixel_values=pv, input_boxes=ib)
        mask = self.conv(samoutput["pred_masks"])[:,0,0,:,:]
        op = torch.sum(mask, dim=(1, 2)) / 256 / 9
        return torch.reshape(op, (op.shape[0], 1))
    
    def mask(self, pv, ib):
        samoutput = self.sam(pixel_values=pv, input_boxes=ib)
        ans = self.conv(samoutput["pred_masks"])[:,0,0,:,:]
        return ans

class Data(Dataset):
    def __init__(self, rootdir="./images", annotationdir="./annotation.json", samtype="facebook/sam-vit-base", device="cuda", filelist=None):
        self.filelist = filelist
        if filelist is None:
            self.filelist = os.listdir(rootdir)
        self.rootdir = rootdir
        self.device = device

        with open(annotationdir, "r") as f:
            self.js = json.load(f)

        self.processor = SamProcessor.from_pretrained(samtype)

    def __len__(self):
        return len(self.filelist)
    
    def __getitem__(self, index):
        fname = self.filelist[index]
        fpath = os.path.join(self.rootdir, fname)

        # read image to numpy array.
        img = cv2.imread(fpath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # get first 3 bound boxes.
        boxes = self.js[fname]['box_examples_coordinates']
        input_boxes = []
        for i in boxes:
            input_boxes.append([i[0][0], i[0][1], i[2][0], i[2][1]])
        
        # get input and load it to device, need to extract one dimention.
        imgtensor = self.processor(img, return_tensors="pt", input_boxes=[input_boxes[:3]]).to(self.device)

        # print(imgtensor)

        label = torch.tensor([len(self.js[fname]["points"])], dtype=torch.float).to(self.device)

        return imgtensor["pixel_values"][0], imgtensor["input_boxes"][0], label, fname

def splitdata(rootdir="./images", trainscale=0.8):
    ftrain = "./train.txt"
    fvalid = "./valid.txt"
    flist = os.listdir(rootdir)
    trainsz = int(trainscale * len(flist))

    splitted = False

    if os.path.isfile(ftrain):
        with open(ftrain, "r") as f:
            if len(f.read().split()) == trainsz:
                splitted = True
    
    if splitted:
        f1 = open(ftrain, "r")
        f2 = open(fvalid, "r")
        return f1.read().split(), f2.read().split()
    
    np.random.shuffle(flist)
    f1 = open(ftrain, "w")
    f2 = open(fvalid, "w")
    f1.write("\n".join(flist[:trainsz]))
    f2.write("\n".join(flist[trainsz:]))

    return flist[:trainsz], flist[trainsz:]
    
def main():
    device = "cuda"
    savedmodel = "./vit-sam-changed-l1loss.pth"
    bsz = 8

    vit_lin = ViTSum()
    if os.path.isfile(savedmodel):
        vit_lin.load_state_dict(torch.load(savedmodel))
    vit_lin = vit_lin.to(device=device)
    
    ftrain, fvalid = splitdata()

    traindata = Data(filelist=ftrain)
    validdata = Data(filelist=fvalid)
    
    trainloader = DataLoader(traindata, batch_size=1, shuffle=True, drop_last=True)
    validloader = DataLoader(validdata, batch_size=1, shuffle=False, drop_last=False)

    loss = nn.L1Loss().to(device=device)
    optm = torch.optim.Adam(params=vit_lin.parameters(), lr=1e-5)

    # imgs, boxes, label
    for i in range(10):
        s = 0
        batch = 0
        print("train start!")
        for pv, ib, label, fname in trainloader:
            y = vit_lin(pv, ib)
            ls = loss(label, y)
            ls.backward()
            s += ls.item()
            batch += 1
            print(y)
            print(label)
            print(f"{batch} finished, loss {ls.item()}")
            if batch % bsz == 0:
                optm.step()
                optm.zero_grad()
        torch.save(vit_lin.state_dict(), savedmodel)
        print(i, s)
    return 0

def randomselect():
    device = "cuda"
    savedmodel = "./vit-sam-changed-l1loss.pth"

    vit_lin = ViTSum(drop=0)
    if os.path.isfile(savedmodel):
        vit_lin.load_state_dict(torch.load(savedmodel))
    vit_lin = vit_lin.to(device=device)

    for parms in vit_lin.parameters():
        parms.requires_grad = False
    
    ftrain, fvalid = splitdata()

    traindata = Data(filelist=ftrain)
    validdata = Data(filelist=fvalid)
    
    trainloader = DataLoader(traindata, batch_size=1, shuffle=True, drop_last=False)
    validloader = DataLoader(validdata, batch_size=1, shuffle=True, drop_last=False)

    cnt = 0
    for pv, ib, label, fname in trainloader:
        cnt += len(label)
        y = vit_lin.mask(pv, ib).cpu().numpy()
        print(y)
        names = "_".join([i.split(".")[0] for i in fname])
        np.save(f"./encoder/{names}.npy", y)
        if cnt >= 16:
            break
    return 0

def accuracy(accset="train"):
    device = "cuda"
    savedmodel = "./vit-sam-changed-l1loss.pth"

    vit_lin = ViTSum(drop=0)
    if os.path.isfile(savedmodel):
        vit_lin.load_state_dict(torch.load(savedmodel))
    vit_lin = vit_lin.to(device=device)

    vit_lin.eval()
    for parms in vit_lin.parameters():
        parms.requires_grad = False
    
    ftrain, fvalid = splitdata()

    traindata = Data(filelist=ftrain)
    validdata = Data(filelist=fvalid)
    
    trainloader = DataLoader(traindata, batch_size=1, shuffle=False, drop_last=False)
    validloader = DataLoader(validdata, batch_size=1, shuffle=False, drop_last=False)

    # imgs, boxes, label
    mae = 0
    rmse = 0
    cnt = 0
    for pv, ib, label, fname in trainloader:
        cnt += len(label)
        y = vit_lin(pv, ib)
        # print(label.item(), y.item())
        mae += torch.sum(torch.abs(label - y)).item()
        rmse += torch.sum((label - y) ** 2).item()
        print(cnt, mae / cnt, (rmse / cnt) ** 0.5)
    
    ans1 = [mae / cnt, (rmse / cnt) ** 0.5]

    mae = 0
    rmse = 0
    cnt = 0

    if accset == "train":
        validloader = trainloader
    for pv, ib, label, fname in validloader:
        cnt += len(label)
        y = vit_lin(pv, ib)
        # print(label.item(), y.item())
        mae += torch.sum(torch.abs(label - y)).item()
        rmse += torch.sum((label - y) ** 2).item()
        print(cnt, mae / cnt, (rmse / cnt) ** 0.5)

    ans2 = [mae / cnt, (rmse / cnt) ** 0.5]

    print(ans1)
    print(ans2)
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='model test')
    parser.add_argument('--train', type=bool,
                    help='Whether to train the model')
    parser.add_argument('--randomselect', type=bool,
                    help='Whether to randomly select som images to process')
    parser.add_argument('--accuracy', type=str,
                    help='test the train set and valid set accuracy')
    
    args = parser.parse_args()
    print(args.train)
    if args.train:
        main()
    if args.randomselect:
        randomselect()
    if args.accuracy:
        accuracy(accset=args.accuracy)
    # main()
    # randomselect()
    # accuracy()