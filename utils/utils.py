import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class ImageFolderDataset(Dataset):
    def __init__(self,root,transform=None):
        super(ImageFolderDataset,self).__init__()
        self.root=root
        self.tansform=transform
        self.file=list(os.listdir(self.root))
        self.file=[p for p in self.file if p.endswith(('.jpg','.png','.jpeg'))]
        
    
    def __len__(self):
        return len(self.file)
    
    def __getitem__(self, idx):
        image_path=os.path.join(self.root,self.file[idx])
        image=Image.open(image_path).convert('RGB')
        
        if self.tansform:
            image=self.tansform(image)
            
        return image
    

def get_transform(size,crop,final_size):
    transform_list=[]
    if size > 0:
        transform_list.append(transforms.Resize(size))
    
    if crop:
        transform_list.append(transforms.Resize(final_size))
        transform_list.append(transforms.RandomCrop(final_size))
    
    else:
        transform_list.append(transforms.Resize(final_size))
        
    transform_list.append(transforms.ToTensor())
    
    return transforms.Compose(transform_list)

def adaptive_instance_normalication(content_feat,style_feat):
    # [batch size,chanels,h,w]
    size=content_feat.size()
    style_mean,style_std=calc_mean_std(style_feat)
    content_mean,cotent_std=calc_mean_std(content_feat)
    
    normalized_content_feat=(content_feat-content_mean.expand(size))/cotent_std.expand(size)
    
    return normalized_content_feat*style_std.expand(size) + style_mean.expand(size)

def calc_mean_std(feat,eps=1e-5):
    # [batch size,chanels,h,w]
    size=feat.size()
    assert (len(size)==4)
    batch_size,chanels=size[:2]
    feat_mean=feat.view(batch_size,chanels,-1).mean(dim=2).view(batch_size,chanels,1,1)
    feat_var=feat.view(batch_size,chanels,-1).var(dim=2,unbiased=False) + eps
    feat_std=feat_var.sqrt().view(batch_size,chanels,1,1)
    return feat_mean,feat_std
         