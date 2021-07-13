from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from utils import save_best_record
from model import Model
from dataset import Dataset
from train import train
from test_10crop import test
import option
from tqdm import tqdm
from utils import Visualizer
from config import *
import sys

viz = Visualizer(env='Child Abuse Detection', use_incoming_socket=False)

if __name__ == '__main__':
    args = option.parser.parse_args()
    config = Config(args)
    train_nloader = DataLoader(Dataset(args, test_mode=False, is_normal=True),
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=False, drop_last=True)
    train_aloader = DataLoader(Dataset(args, test_mode=False, is_normal=False),
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=False, drop_last=True)
    test_loader = DataLoader(Dataset(args, test_mode=True),
                              batch_size=1, shuffle=False,  ####
                              num_workers=0, pin_memory=False)
    #print(len(train_nloader))
    #print(len(train_aloader))
    #print(len(test_loader))
    #sys.exit()
    #for i in enumerate(test_loader):
    #    print(i)

    model = Model(args.feature_size, args.batch_size)
    if args.pretrained_ckpt is not None:
        checkpoint = torch.load(args.pretrained_ckpt)
        model.load_state_dict(checkpoint['model_state_dict'])
        #model.load_state_dict(torch.load(args.pretrained_ckpt))
        #model.eval()

    for name, value in model.named_parameters():
        print(name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if not os.path.exists('./ckpt'):
        os.makedirs('./ckpt')

    optimizer = optim.Adam(model.parameters(),
                            lr=config.lr[0], weight_decay=0.005)
    if args.pretrained_ckpt is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    test_info = {"epoch": [], "test_AUC": []}
    best_AUC = -1
    output_path = ''   # put your own path here
    auc = test(test_loader, model, args, viz, device, best_AUC)

    for step in tqdm(
            range(1, args.max_epoch + 1),
            total=args.max_epoch,
            dynamic_ncols=True
    ):
        if step > 1 and config.lr[step - 1] != config.lr[step - 2]:
            for param_group in optimizer.param_groups:
                param_group["lr"] = config.lr[step - 1]

        if (step - 1) % len(train_nloader) == 0:
            loadern_iter = iter(train_nloader)
        if (step - 1) % len(train_aloader) == 0:
            loadera_iter = iter(train_aloader)
        
        train(loadern_iter, loadera_iter, model, args.batch_size, optimizer, viz, device)

        if step % 5 == 0 and step > 0:
            auc = test(test_loader, model, args, viz, device, best_AUC)
            test_info["epoch"].append(step)
            test_info["test_AUC"].append(auc)
            if test_info["test_AUC"][-1] > best_AUC:
                best_AUC = test_info["test_AUC"][-1]
                torch.save({'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),'epoch':step}, './ckpt/' + args.model_name + '.pkl')
                save_best_record(test_info, os.path.join(output_path, 'AUC-previous.txt'))
    #torch.save(model.state_dict(), './ckpt/' + args.model_name + 'final.pkl')

