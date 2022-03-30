import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.accs=0
        self.F1=0
        self.F2 = 0
        self.F3 = 0
        self.F4 = 0
        self.val_loss_min = np.Inf
        self.actual_true__best_acc = None
        self.actual_true__best_acc_dict = None
        self.actual_true__best_acc_epoch = None
        self.actual_true__best_f1 = None
        self.actual_true__best_f1_dict = None
        self.actual_true__best_f1_epoch = None

    def __call__(self, val_loss, accs,F1,F2,F3,F4,model,modelname,str,rawdict_inp,epoch):
        try:
            actual_true__true_precision = rawdict_inp[1]["TP"] /(rawdict_inp[1]["TP"]+rawdict_inp[1]["FP"])
        except ZeroDivisionError:
            actual_true__true_precision = 0
        try:
            actual_true__true_recall = rawdict_inp[1]["TP"] /(rawdict_inp[1]["TP"]+rawdict_inp[1]["FN"])
        except ZeroDivisionError:
            actual_true__true_recall = 0
        try:
            actual_true__true_f1 = (2*actual_true__true_precision*actual_true__true_recall)/(actual_true__true_precision+actual_true__true_recall)
        except ZeroDivisionError:
            actual_true__true_f1 = 0  # it's... 0 f1.
            
        actual_true__true_acc = (rawdict_inp[1]["TP"]+rawdict_inp[1]["TN"]) /(rawdict_inp[1]["TP"]+rawdict_inp[1]["FN"]+rawdict_inp[1]["TN"]+rawdict_inp[1]["FP"])
        
        if self.actual_true__best_acc==None or self.actual_true__best_acc<actual_true__true_acc: # beat.
            self.actual_true__best_acc = actual_true__true_acc
            self.save_checkpoint(model,modelname,str+"_bestacc")
            self.actual_true__best_acc_dict = rawdict_inp
            self.actual_true__best_acc_epoch = epoch
            
        if self.actual_true__best_f1==None or self.actual_true__best_f1<actual_true__true_f1:
            self.actual_true__best1 = actual_true__true_f1
            self.save_checkpoint(model,modelname,str+"_bestf1")
            self.actual_true__best_f1_dict = rawdict_inp
            self.actual_true__best_f1_epoch = epoch
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.accs = accs
            self.F1 = F1
            self.F2 = F2
            self.F3 = F3
            self.F4 = F4
            self.val_loss_min = val_loss
            self.save_checkpoint(model,modelname,str)
        elif score < self.best_score:
            self.counter += 1
            # print('EarlyStopping counter: {} out of {}'.format(self.counter,self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
                print("BEST Accuracy: {:.4f}|NR F1: {:.4f}|FR F1: {:.4f}|TR F1: {:.4f}|UR F1: {:.4f}"
                      .format(self.accs,self.F1,self.F2,self.F3,self.F4))
                print("Actual Best Accuracy(after fix):",self.actual_true__best_acc)
                print("Actual Best Accuracy Details (after fix):",self.actual_true__best_acc_dict)
                print("Actual Best Accuracy epoch number (after fix):",self.actual_true__best_acc_epoch)
                print("Actual Best F1 (after fix):",self.actual_true__best_f1)
                print("Actual Best F1 Details (after fix):",self.actual_true__best_f1_dict)
                print("Actual Best F1 epoch number (after fix):",self.actual_true__best_f1_epoch)

        else: # beat score
            self.best_score = score
            self.accs = accs
            self.F1 = F1
            self.F2 = F2 
            self.F3 = F3
            self.F4 = F4
            self.val_loss_min = val_loss
            self.save_checkpoint(model,modelname,str)
            self.counter = 0

    def save_checkpoint(self,model,modelname,str):
        '''Saves model when validation loss decrease.'''
        # if self.verbose:
        #     print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(self.val_loss_min,val_loss))
        torch.save(model.state_dict(),modelname+str+'.m')
