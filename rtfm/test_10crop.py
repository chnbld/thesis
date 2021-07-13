import matplotlib.pyplot as plt
import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import numpy as np

def test(dataloader, model, args, viz, device):
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0)
        
        myfile=open('output-rtfm.txt','w')
        #dataloader=enumerate(dataloader)
        for i,input in enumerate(dataloader):
            input = input.to(device)
            #print(input.shape)
            input = input.permute(0,2,1,3)
            #print(f'testcrop input_________________{input.shape}')
            score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_abn_bottom, feat_select_normal_bottom, logits,a1,a2,_ = model(inputs=input)
            #logits = model(inputs=input)

            logits = torch.squeeze(logits, 1)
            logits = torch.mean(logits, 0)

            sig = logits
            myfile.writelines(f'i={i+1}')
            for l in sig:
                myfile.writelines(np.array_str(l.detach().cpu().numpy()))
            
        myfile.close()
            #pred = torch.cat((pred, sig))

        #gt = np.load(args.gt)
        #pred = list(pred.cpu().detach().numpy())
        #pred = np.repeat(np.array(pred), 16)
        #for l in pred:
        #    print(l)

        #fpr, tpr, threshold = roc_curve(list(gt), pred)
        #np.save('fpr.npy', fpr)
        #np.save('tpr.npy', tpr)
        #rec_auc = auc(fpr, tpr)
        #if rec_auc>best_AUC:
        #    np.save('fpr-rtfm.npy', fpr)
        #    np.save('tpr-rtfm.npy', tpr)
        #print('auc : ' + str(rec_auc))

        #precision, recall, th = precision_recall_curve(list(gt), pred)
        #pr_auc = auc(recall, precision)
        #np.save('precision.npy', precision)
        #np.save('recall.npy', recall)
        #viz.plot_lines('pr_auc', pr_auc)
        #viz.plot_lines('auc', rec_auc)
        #viz.lines('scores', pred)
        #viz.lines('roc', tpr, fpr)
        #return rec_auc
        return 0 

