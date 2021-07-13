import torch.utils.data as data
import numpy as np
from utils import process_feat
import torch
from torch.utils.data import DataLoader
torch.set_default_tensor_type('torch.cuda.FloatTensor')


class Dataset(data.Dataset):
    def __init__(self, args, is_normal=True, transform=None, test_mode=False):
        self.modality = args.modality
        self.is_normal = is_normal
        if test_mode:
            self.rgb_list_file = args.test_rgb_list
        else:
            self.rgb_list_file = args.rgb_list
        self.tranform = transform
        self.test_mode = test_mode
        self._parse_list()
        self.num_frame = 0
        self.labels = None

    def _parse_list(self):
        self.list = list(open(self.rgb_list_file))
        if self.test_mode is False:
            if self.is_normal:
                #ca
                self.list = self.list[:99]
                #ucf
                #self.list = self.list[81:]
                print('normal list')
                print(self.list)
            else:
                #ca
                self.list = self.list[99:]
                #ucf
                #self.list = self.list[:81]

                print('abnormal list')
                print(self.list)

    def __getitem__(self, index):

        label = self.get_label(index) # get video level label 0/1
        #print(self.list[index].strip('\n'))
        features = np.load(self.list[index].strip('\n'), allow_pickle=True)
        features = np.array(features, dtype=np.float32)
        #print(features.shape)
        features = features[:,None,:]
        #features = np.transpose(features,(1,0,2))
        #print(features.shape)

        if self.tranform is not None:
            features = self.tranform(features)
        if self.test_mode:
            return torch.from_numpy(features)
        else:
            features = features.transpose(1, 0, 2)  # [10, B, T, F]
            divided_features = []
            for feature in features:
                f1,f2=np.split(feature,2)
                feature = process_feat(f1, 32)
                feature2 = process_feat(f2,32)
                #print(f'feature2:{feature2.shape}')
                feature=np.concatenate((feature,feature2),axis=0)
                #print(f'feature:{feature.shape}')
                divided_features.append(feature)
            divided_features = np.array(divided_features, dtype=np.float32)
            
            #print(divided_features.shape)
            return divided_features, label

    def get_label(self, index):
        if self.is_normal:
            # label[0] = 1
            label = torch.tensor(0.0)
        else:
            label = torch.tensor(1.0)
            # label[1] = 1
        return label

    def __len__(self):
        return len(self.list)

    def get_num_frames(self):
        return self.num_frame