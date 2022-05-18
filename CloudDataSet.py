from numpy.lib.type_check import imag
from tifffile.tifffile import imread
import torch
import numpy as np
import csv
import cv2
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
from PIL import Image
import math
import os
import tifffile
import random
from config import cfg
root=""
#B,G,R,NIR,SWIR,SWIR,CIRUS
L8_BANDS=['B2','B3','B4','B5','B6','B7','B9','mask']
L8_Base_Dir='/data/xuxinyi/Cloud-Datasets/LandSat8/LandSat8-biome-shadow'
#B,G,R.NIR,
S2_BANDS=[2,3,4,8,12,13]
#Usually only use B,G,R.
L8_RGB_BANDS=['B2','B3','B4','mask']
S2_RGB_BANDS=[2,3,4]

PATCH_SIZE=512
class LandSat8Biome(Dataset):
    def __init__(self,base_dir=L8_Base_Dir,bands=L8_RGB_BANDS,
            datatype='train',shadow=False,nclass=2,patchsize=PATCH_SIZE,
            transform=False,mean=[],std=[],**kwargs):
      
        #The element of the images_list is images_ dict
        #There are three elements in each dict:
        # band_ Dir ': the path of the band patch in;
        #' band_ csv_ Path 'is the CSV file of patchname
        # band_patches:The name of the patches corresponding to each band by reading 'band_csv_Path'
        images_list=[]
        self.name='LandSat8-Biome-'+str(patchsize)
        self.datatype=datatype
        self.shadow=shadow
        self.patch_size=patchsize
        self.nclass=nclass
        self.bands=bands
        self.base_dir=os.path.join(base_dir,str(patchsize))
        if 'csv_path' in kwargs:
            self.csv_path=kwargs['csv_path']
        else:
            self.csv_path=self.base_dir+"/"+datatype+"_patch_name.csv"
        #If Need transform to RGB image
        if 'Transformto255' in kwargs:
            self.Transformto255=kwargs['Transformto255']
        else:
            self.Transformto255=False

        with open(self.csv_path) as f:
            reader=csv.reader(f)
            reader=list(reader)
        if len(reader[0])<2:
            del reader[0]
        self.patch_list=sum(reader,[])
        if 'choice_num' in kwargs:
            self.patch_list=random.sample(self.patch_list,kwargs['choice_num'])
        self.transform=transform
        
        self.mean=mean
        self.std=std
        if self.shadow:
            if self.nclass==3:
                #With Cloud shadow, 3 classes
                #0-0 filling, 128-0 cloud free, 64-1 cloud shadow, 192-2 thin cloud, 255-2 cloud
                self.class_names=['0:Clear','1:Shadow','2:Thin&Thick Cloud',]
            else:
                #With Cloud shadow，4 classes
                #0-0filling，128-0 cloud free，64-1 cloud shadow，192-2 thin cloud，255-3 Cloud
                self.class_names=['0:Clear','1:Shadow','2:Thin Cloud','3:Cloud']
        else:
            if self.nclass==2:
                #Without Cloud Shadow, 2classes
                #0-0 filling，128-0 cloud free，64-0 cloud shadow，192-2 thin cloud，255-2 cloud
                self.class_names=['0:Clear','1:Thin&Thick Cloud']
            else:
                #Without Cloud Shadow, 3classes
                #0-0 filling，128-0 cloud free，64-0 cloud shadow，192-2 thin cloud，255-2 cloud
                self.class_names=['0:Clear','1:Thin Cloud','2:Cloud']
                        
        

    def __getitem__(self, index: int):
        image=[]
        mask=np.empty((self.patch_size, self.patch_size))
        mask.fill(255)
        single_image_name=''
        for band in self.bands:
            band_patch_name=self.patch_list[index]+"_"+band+".tif"
            band_patch_dir=os.path.join(self.base_dir,band)
            band_patch_path=os.path.join(band_patch_dir,band_patch_name)
            img=np.array(tifffile.imread(band_patch_path,0))
            if band=='mask':
                if self.shadow:
                    if self.nclass==3:
                         #With Cloud shadow, 3 classes
                          #0-0 filling, 128-0 cloud free, 64-1 cloud shadow, 192-2 thin cloud, 255-2 cloud
                        mask[img==0]=0
                        mask[img==128]=0
                        mask[img==64]=1
                        mask[img==192]=2
                        mask[img==255]=2
                    else:
                         #With Cloud shadow，4 classes
                         #0-0filling，128-0 cloud free，64-1 cloud shadow，192-2 thin cloud，255-3 Cloud
                        mask[img==0]=0
                        mask[img==128]=0
                        mask[img==64]=1
                        mask[img==192]=2
                        mask[img==255]=3
                        
                else:
                    if self.nclass==2:
                        #Without Cloud Shadow, 2classes
                        #0-0 filling，128-0 cloud free，64-0 cloud shadow，192-2 thin cloud，255-2 cloud
                        mask[img==0]=0
                        mask[img==128]=0
                        mask[img==64]=0
                        mask[img==192]=1
                        mask[img==255]=1
                    else:
                         #Without Cloud Shadow, 3classes
                        #0-0 filling，128-0 cloud free，64-0 cloud shadow，192-2 thin cloud，255-2 cloud
                        mask[img==0]=0
                        mask[img==128]=0
                        mask[img==64]=0
                        mask[img==192]=1
                        mask[img==255]=2
                mask=mask.astype(np.float32)
            else:
                image.append(img.astype(np.float32))
        image=np.array(image)
        image=torch.tensor(image)
        if self.transform and self.mean!=[] and self.std!=[]:
            image=transforms.Normalize(self.mean,self.std)(image)
        mask=torch.tensor(mask)
        #If Need transform to RGB image
        if self.Transformto255:
            image=to255(image)

        return image,mask,self.patch_list[index]

    def __len__(self):
        return len(self.patch_list)

class Sentinel2CloudMaskCatalogue(Dataset):
    def __init__(self,base_dir,bands=S2_RGB_BANDS,datatype='train',
                shadow=False,nclass=2,patchsize=PATCH_SIZE,
                transform=False,mean=[],std=[],**kwargs):
        if bands==None:
            self.bands= [2,3,4,8,12,13]
        self.name="Sentinel-2 Cloud Mask Catalogue"
        self.img_dir=os.path.join(base_dir,'img')
        self.mask_dir=os.path.join(base_dir,'mask')
        self.shadow=shadow
        self.nclass=nclass
        if 'csv_path' in kwargs:
            self.csv_path=kwargs['csv_path']
        else:
            self.csv_path=self.base_dir+"/"+datatype+".csv"
        if 'Transformto255' in kwargs:
            self.Transformto255=kwargs['Transformto255']
        else:
            self.Transformto255=False
        with open(self.csv_path) as f:
            reader=csv.reader(f)
            reader=list(reader)
        if len(reader[0])<2:
            del reader[0]
        self.name_list=sum(reader,[])
        f.close()
        self.transform=transform
        if 'choice_num' in kwargs:
            self.name_list=random.sample(self.name_list,kwargs['choice_num'])
        self.bands=bands
        self.mean=mean
        self.std=std
        if self.shadow:
            if self.nclass==3:
                #With shadow，3 class
                #0-0 filling, 128-0 cloud free, 64-1 cloud shadow, 192-2 thin cloud, 255-2 cloud
                self.class_names=['0:Clear','1:Shadow','2:Thin&Thick Cloud',]
            else:
                #With shadow，4 class
                #0-0 filling, 128-0 cloud free, 64-1 cloud shadow, 192-2 thin cloud, 255-3 cloud
                self.class_names=['0:Clear','1:Shadow','2:Thin Cloud','3:Cloud']
        else:
            if self.nclass==2:
                #Without Shadow ,2 classes
                #0-0 filling, 128-0 cloud free,, 192-1 thin cloud, 255-1 cloud
                self.class_names=['0:Clear','1:Thin&Thick Cloud']
            else:
                #Without Shadow ,3 classes
                #0-0filling，128-0  cloud free，64-0Cloud Shadow，192-2thin cloud，255-2 Cloud
                self.class_names=['0:Clear','1:Thin Cloud','2:Cloud']

    def __getitem__(self, index: int):
        name=self.name_list[index]
        img_path=os.path.join(self.img_dir,name)
        mask_path=os.path.join(self.mask_dir,name)
        img_tmp=np.load(img_path,encoding = "latin1")
        mask_tmp=np.load(mask_path,encoding = "latin1")
        img=np.empty([len(self.bands),mask_tmp.shape[0],mask_tmp.shape[1]])
        
        for i in range(len(self.bands)):
            #The data content is the original reflectance divided by 10000
            img[i,:,:]=img_tmp[:,:,(self.bands[i]-1)]
         
        image=torch.Tensor(img*10000)
        mask=torch.Tensor(mask_tmp)
        if self.transform and self.mean!=[] and self.std!=[]:
            image=transforms.Normalize(self.mean,self.std)(image)
        if self.Transformto255:
            image=to255(image)

        return image,mask,name

    def __len__(self):
        return len(self.name_list)



class LandSat8Sentinel2Data(Dataset):
    def __init__(self,L8BaseDir,L8CsvPath,L8Mean,
                        L8Std,S2BaseDir,S2CsvPath,S2Mean,S2Std,choice_num,
                                transform=True,shadow=False,nclass=2,Transformto255=True):
        self.len=choice_num
        self.Transformto255=Transformto255
        self.LandSat8Data=LandSat8Biome(base_dir=L8BaseDir,transform=transform,mean=L8Mean,std=L8Std,shadow=shadow,nclass=nclass,csv_path=L8CsvPath,choice_num=choice_num,Transformto255=Transformto255)
        self.Sentinel2Data=Sentinel2CloudMaskCatalogue(base_dir=S2BaseDir,transform=transform,mean=S2Mean,std=S2Std,shadow=shadow,nclass=nclass,csv_path=S2CsvPath,choice_num=choice_num,Transformto255=Transformto255)
    def __getitem__(self,index:int):
        L8=list(self.LandSat8Data.__getitem__(index))
        S2=list(self.Sentinel2Data.__getitem__(index))

        return {'A':L8,'B':S2}
    def __len__(self):
        return self.len



if __name__=='__main__':

    dataset=LandSat8Sentinel2Data(L8BaseDir=cfg.SOURCE_TRAIN_DIR,
                                    L8CsvPath=cfg.SOURCE_TRAIN_LIST,
                                    L8Mean=cfg.TRAIN.SOURCE_TRAIN_MEAN,
                                    L8Std=cfg.TRAIN.SOURCE_TRAIN_STD,
                                    S2BaseDir=cfg.TARGET_DIR,
                                    S2CsvPath=cfg.TARGET_TRAIN_LIST,
                                    S2Mean=cfg.TRAIN.TARGET_MEAN,
                                    S2Std=cfg.TRAIN.TARGET_STD,
                                    choice_num=10000,
                                    transform=False,
                                    to255=True
                                    )
    dataloader=DataLoader(dataset,batch_size=2)
    i=0
    print(dataset.__len__())
    for data in dataloader:
        A=data['A']
        print(A[2])
        i=i+1
        if i>2:
            break

def to255(image):
    #image：(C,H,W)
    image=np.array(image.permute(1,2,0))
    image_max=np.max(np.max(image,axis=0),axis=0)
    image_min=np.min(np.min(image,axis=0),axis=0)
    image=(image-image_min)*255/(image_max-image_min)
    image=torch.tensor(image.round()).permute(2,0,1)

    return image
