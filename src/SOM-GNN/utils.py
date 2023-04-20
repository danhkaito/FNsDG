import torch
import random
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, precision_recall_fscore_support
import torch.nn.functional as F



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_class_acc(output, labels):
    #print class-wise performance
    '''
    for i in range(labels.max()+1):
        
        cur_tpr = accuracy(output[pre_num:pre_num+class_num_list[i]], labels[pre_num:pre_num+class_num_list[i]])
        print(str(pre)+" class {:d} True Positive Rate: {:.3f}".format(i,cur_tpr.item()))

        index_negative = labels != i
        labels_negative = labels.new(labels.shape).fill_(i)
        
        cur_fpr = accuracy(output[index_negative,:], labels_negative[index_negative])
        print(str(pre)+" class {:d} False Positive Rate: {:.3f}".format(i,cur_fpr.item()))

        pre_num = pre_num + class_num_list[i]
    '''

    #ipdb.set_trace()
    auc_score = roc_auc_score(labels, F.softmax(output, dim=-1)[:,1].detach().cpu(), average='macro')

    pred_labels = torch.argmax(output, dim=-1).detach().cpu().numpy()
    macro_F1 = f1_score(labels, pred_labels, average='macro')
    micro_F1 = f1_score(labels, pred_labels, average='micro')
    print(f"Accuracy: {micro_F1}\n \
          Macro_F1: {macro_F1}\n\
          AUC-ROC: {auc_score}\n\
          Precion, Recall, F1-Score Macro: {precision_recall_fscore_support(labels, pred_labels, average='macro')}\n\
          Precsion, Recall, F1-Score Label 1: {precision_recall_fscore_support(labels, pred_labels, average='binary', pos_label = 1)}\n\
          Precsion, Recall, F1-Score Label 0: {precision_recall_fscore_support(labels, pred_labels, average='binary', pos_label = 0)}")
    return