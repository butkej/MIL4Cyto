'''
This submodule contains miscellaneous functions to compute and analyze metrics. 
Used in custom_model.py and main.py
'''

# IMPORTS
#########

import itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, auc, roc_curve
import torch

plt.style.use('ggplot')
# FUNCTIONS
###########

# METRICS

def binary_accuracy(outputs, targets):
    assert targets.size() == outputs.size()
    y_prob = torch.ge(outputs, 0.5).float()
    return (targets == y_prob).sum().item() / targets.size(0)

# PLOTS

def reconstruct_image(img, coords, attention=None, background=False):
    ''' Reconstructs a given whole slide image (img) but only showing the extracted patches.
    These patches are used in MIL and can be further modified by attention values when a 
    vector/list of attention weights is passed.
    Boolean 'background' setting: If False rest of image is black, if True rest of img is
    scaled down to 10%.
    
    Returns the 'reconstructed image' with bounding boxes around the patches.
    '''
    # build empty black base img
    img_shape = img.shape
    if background == False:
        reconst_img = np.zeros(shape=(img_shape[0],img_shape[1]), dtype=np.uint8)
        
    elif background == True: # alternativly use the original image downscaled in intensity
        reconst_img = img * 0.1
        
    # now fill with relevant patches
    if attention is not None:
        assert len(coords) == len(attention)
        # rescale the weights
        attention_min = attention.min()
        attention_max = attention.max()
        for patch, patch_attention in zip(coords, attention):
            patch_attention_scaled = (patch_attention - attention_min) / (attention_max - attention_min)
            y_left, y_right, x_left, x_right = patch
            reconst_img[y_left:y_right, x_left:x_right] = img[y_left:y_right, x_left:x_right]
            reconst_img[y_left:y_right, x_left:x_right] = reconst_img[y_left:y_right, x_left:x_right] * patch_attention_scaled
            cv2.rectangle(reconst_img,(x_left, y_left), (x_right, y_right), (255,255,255), 3)
    else:
        for patch in coords:
            y_left, y_right, x_left, x_right = patch
            reconst_img[y_left:y_right, x_left:x_right] = img[y_left:y_right, x_left:x_right]
            cv2.rectangle(reconst_img,(x_left, y_left), (x_right, y_right), (255,255,255), 3)
    
    return reconst_img

def plot_accuracy(history, savepath):
    ''' takes a history object and plots the accuracies
    '''
    train_acc = [i['train_acc'] for i in history]
    val_acc = [x['val_acc'] for x in history]
    plt.plot(train_acc)
    plt.plot(val_acc)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['Training', 'Validation'])
    plt.title('Accuracy vs. No. of epochs for training and validation')
    plt.savefig(savepath +'_accuracies.pdf', dpi=600)
    plt.clf()

def plot_losses(history, savepath):
    ''' takes a history object and plots the losses
    '''
    train_loss = [i['train_loss'] for i in history]
    val_loss = [x['val_loss'] for x in history]
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs for training and validation')
    plt.savefig(savepath +'_losses.pdf', dpi=600)
    plt.clf()

def plot_conf_matrix(y_true, y_pred, savepath,
        target_names,
        title='Confusion Matrix',
        normalize=True):
    '''computes and plots the confusion matrix using sklearn
    Title can be set arbitrarily but target_names should be a list of class names eg. ['positive', 'negative']
    '''
    conf_mat = confusion_matrix(y_true, y_pred)

    if len(target_names) == 2:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        binary_classification_counts = list((tn, fp, fn, tp))
        print('TN, FP, FN, TP')
        print(binary_classification_counts)

    acc = np.trace(conf_mat) / float(np.sum(conf_mat))
    misclass = 1 - acc

    cmap = plt.get_cmap('Blues')

    if normalize:
        conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
        title = title + ' (Normalized)'

    #plt.figure(figsize=(8,7))
    plt.imshow(conf_mat, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    plt.grid(False)

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    plt.title(title)

    thresh = conf_mat.max() / 1.5 if normalize else conf_mat.max() / 2
    for i, j in itertools.product(range(conf_mat.shape[0]), range(conf_mat.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(conf_mat[i, j]),
                     horizontalalignment="center",
                     color="white" if conf_mat[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(conf_mat[i, j]),
                     horizontalalignment="center",
                     color="white" if conf_mat[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(acc, misclass))
    plt.tight_layout()

    plt.savefig(savepath + '_confusion_matrix.pdf', dpi=600)
    plt.clf()
    

def binary_roc_curve(y_true, y_hat_scores):
    ''' Only works for the binary classfication task.
    y_hat_scores are the raw sigmoidal network output probabilities
    (no torch.ge thresholding)

    Returns false positive rate, true positive rate and thresholds
    '''
    fpr, tpr, thresholds = roc_curve(y_true, y_hat_scores)
    return fpr, tpr, thresholds

def plot_binary_roc_curve(fpr, tpr, savepath):
    ''' plots a ROC curve with AUC score
    in a binary classification setting
    '''
    area = auc(fpr, tpr)

    lw = 2
    plt.plot([0, 1], [0, 1], color='blue', lw=lw, linestyle='--') 
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (AUC={:0.3f})'.format(area))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc='lower right')
    plt.xlabel('False Positive Rate (1-specificity)')
    plt.ylabel('True Positive Rate (sensitivity)')
    plt.savefig(savepath + '_binary_roc_curve.pdf', dpi=600)
    plt.clf()

