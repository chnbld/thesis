import matplotlib.pyplot as plt
import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import numpy as np

def test(dataloader, model, args, viz, device):
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0)
        
        out_file=open('output-sd.txt','w')
        for i,input in enumerate(dataloader):
            input = input.to(device)
            input = input.permute(0,2,1,3)
            logits = model(inputs=input)

            logits = torch.squeeze(logits, 1)
            logits = torch.mean(logits, 0)

            sig = logits
            out_file.writelines(f'i={i+1}')
            for l in sig:
                out_file.writelines(np.array_str(l.detach().cpu().numpy()))
        out_file.close()

        return 0

