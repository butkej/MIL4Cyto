#!/usr/bin/env python

"""
Main driver script to be called with its respective arguments.
Loads data, applies preprocessing, compiles model(s), trains them and validates/saves them.
Also plots various results.
"""

# IMPORTS
#########

import os
import numpy as np
import argparse
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import auc
import random
import torch
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# project specific imports (repo-internal)
from joshnet import custom_model
from joshnet import preprocessing
from joshnet import mil_metrics
from joshnet import mining


# FUNCTIONS
###########


def parse_args():
    """Parse input arguments.

    Returns:
        args: argparser.Namespace class object
        An argparse.Namespace class object contains experimental hyper-parameters.
    """
    parser = argparse.ArgumentParser(
        prog="Deep Attention MIL",
        description="Trains a deep attention-based multiple instance learning system",
    )

    # General
    parser.add_argument(
        "-m",
        "--model",
        dest="model",
        help="choice of model [string]",
        default="butkej-attention",
        type=str,
    )

    parser.add_argument(
        "-e",
        "--epochs",
        dest="epochs",
        help="amount of epochs to train for [integer]",
        default=100,
        type=int,
    )

    # Model-related settings
    parser.add_argument(
        "-g",
        "--gpu",
        dest="multi_gpu",
        help="choice of gpu amount [integer]",
        default=0,
        type=int,
    )

    parser.add_argument(
        "-o",
        "--optimizer",
        dest="optimizer",
        help="choice of optimizer [string]",
        default="adam",
        type=str,
    )

    parser.add_argument(
        "--use_gated",
        dest="use_gated",
        help="use gated Attention. is False by default [boolean]",
        action="store_true",
    )

    parser.add_argument(
        "--use_max",
        dest="use_max",
        help="use max pooling in BaselineMIL. is False by default [boolean] and uses mean pooling then",
        action="store_true",
    )

    parser.add_argument(
        "--no_bias",
        dest="use_bias",
        help="use bias. is True by default [boolean]",
        action="store_false",
    )

    parser.add_argument(
        "--use_adaptive",
        dest="use_adaptive",
        help="use adaptive weighting of Li et al. MICCAI 2019. is False by default [boolean]",
        action="store_true",
    )

    args = parser.parse_args()
    return args


# PREPROCESSING (calls from preprocessing.py)
###############


def preprocess_data(data, labels, input_dim):
    """Call preprocessing subroutines to process data into tiles and then into labeled MIL bags

    input_dim decides the size of a single tile with shape (x,y,z) and must coincide with network input_dim!!
    """
    check_data(data[0])

    # new patch generation procedure based on contouring and centroids
    reference_img = random.choice(data)
    reference_img = cv2.cvtColor(reference_img, cv2.COLOR_RGB2GRAY)

    tiled_data = []
    for wsi in data:
        wsi_gray = cv2.cvtColor(wsi, cv2.COLOR_RGB2GRAY)
        # wsi_gray = preprocessing.histogram_matching(wsi_gray, reference_img, multichannel=False)
        # wsi_gray = preprocessing.histogram_equalization(wsi_gray, mode='normal')
        thresh = preprocessing.otsu_thresholding(wsi_gray, blurring=True)
        centroids = preprocessing.find_contours_and_centroids(
            thresh, morph_kernel=(10, 10), lower_bound_area=900, upper_bound_area=5000
        )
        patch_collection = preprocessing.blowup_patches(
            wsi_gray, centroids, patch_size=input_dim, multichannel=False
        )
        patch_collection = preprocessing.filter_patches(
            patch_collection, crop=50, factor=1.025
        )
        tiled_data.append(patch_collection)

    check_data(tiled_data[0])

    # one hot encode labels if they are not binary, eg. for everything except sigmoid output layer
    if len(np.unique(labels)) > 2:
        labels = preprocessing.one_hot_encode_labels_sk(labels)

    bags, labels, og_labels = preprocessing.build_bags(tiled_data, labels)

    del tiled_data, data

    return bags, labels, og_labels


# MODEL
#######

# load and init choosen model
def choose_model(selection):
    """Chooses a model from custom_model.py according to the string specifed in the model CLI argument and build with specified args"""
    if str(selection) == "butkej-attention":
        model = custom_model.DeepAttentionMIL(args)
    elif str(selection) == "butkej-baseline":
        model = custom_model.BaselineMIL(args)
    # elif... more models
    else:
        print("Error! Choosen model is unclear")

    return model


def choose_optimizer(selection, model):
    """Chooses an optimizer according to the string specifed in the model CLI argument and build with specified args"""
    if str(selection) == "adam":
        optim = torch.optim.Adam(
            model.parameters(),
            lr=0.001,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0,
            amsgrad=False,
        )
    elif str(selection) == "adadelta":
        optim = torch.optim.Adadelta(
            model.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0
        )
    elif str(selection) == "momentum":
        optim = torch.optim.SGD(
            model.parameters(),
            lr=0.01,
            momentum=0.9,
            dampening=0,
            weight_decay=0,
            nesterov=False,
        )
    else:
        print("Error! Choosen optimizer or its parameters are unclear")

    return optim


def init_he_weights(model):
    """He or Kaiming Uniform weight initialization.
    Used with torch.nn.Module.apply
    """
    if type(model) == torch.nn.Linear:
        torch.nn.init.kaiming_uniform_(model.weight)
        model.bias.data.fill_(0.01)


# TRAINING/SAVING/PLOTTING
##########################

# shuffle data and split into training and valdation set
def shuffle_and_split_data(dataset, train_percentage=0.7):
    """
    Takes a dataset that was converted from bags to batches and shuffles and splits it into two splits (train/val)
    """
    train_percentage_index = int(train_percentage * len(dataset))
    indices = np.arange(len(dataset))
    random.shuffle(indices)
    train_ind, test_ind = np.asarray(indices[:train_percentage_index]), np.asarray(
        indices[train_percentage_index:]
    )

    training_ds = [dataset[i] for i in train_ind]
    validation_ds = [dataset[j] for j in test_ind]

    del dataset

    return training_ds, validation_ds


@torch.no_grad()
def get_predictions(model, dataloader):
    """takes a trained model and validation or test dataloader
    and applies the model on the data producing predictions

    binary version
    """
    model.eval()

    all_y_hats = []
    all_preds = []
    all_true = []
    all_attention = []

    for batch_id, (data, label) in enumerate(dataloader):
        label = label.squeeze()
        bag_label = label[0]
        bag_label = bag_label.cpu()

        y_hat, preds, attention = model(data.to("cuda:0"))
        y_hat = y_hat.squeeze(dim=0)  # for binary setting
        y_hat = y_hat.cpu()
        preds = preds.squeeze(dim=0)  # for binary setting
        preds = preds.cpu()

        all_y_hats.append(y_hat.numpy().item())
        all_preds.append(preds.numpy().item())
        all_true.append(bag_label.numpy().item())
        attention_scores = np.round(attention.cpu().data.numpy()[0], decimals=3)
        all_attention.append(attention_scores)

        print("Bag Label:" + str(bag_label))
        print("Predicted Label:" + str(preds.numpy().item()))
        print("attention scores (unique ones):")
        print(np.unique(attention_scores))
        # print(attention_scores)

        del data, bag_label, label

    return all_y_hats, all_preds, all_true


@torch.no_grad()
def evaluate(model, dataloader):
    """Evaluate model / validation operation
    Can be used for validation within fit as well as testing.
    """
    model.eval()
    test_losses = []
    test_acc = []
    result = {}

    for batch_id, (data, label) in enumerate(dataloader):
        label = label.squeeze()
        bag_label = label[0]
        data = data.to("cuda:0")
        bag_label = bag_label.to("cuda:3")

        loss, attention_weights = model.compute_loss(data, bag_label)
        test_losses.append(float(loss))
        acc = model.compute_accuracy(data, bag_label)
        test_acc.append(float(acc))

        del data, bag_label, label

    result["val_loss"] = sum(test_losses) / len(test_losses)
    result["val_acc"] = sum(test_acc) / len(test_acc)
    return result, attention_weights


def fit(model, optim, train_dl, validation_dl, model_savepath):
    """Trains a model on the previously preprocessed train and val sets.
    Also calls evaluate in the validation phase of each epoch.
    """
    best_acc = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        # TRAINING PHASE
        model.train()
        train_losses = []
        train_acc = []

        for batch_id, (data, label) in enumerate(train_dl):
            label = label.squeeze()
            bag_label = label[0]

            data = data.to("cuda:0")
            bag_label = bag_label.to("cuda:3")

            model.zero_grad()  # resets gradients

            loss, _ = model.compute_loss(data, bag_label)  # forward pass
            train_losses.append(float(loss))
            acc = model.compute_accuracy(data, bag_label)
            train_acc.append(float(acc))

            loss.backward()  # backward pass
            optim.step()  # update parameters
            # optim.zero_grad() # reset gradients (alternative if all grads are contained in the optimizer)
            # for p in model.parameters(): p.grad=None # alternative for model.zero_grad() or optim.zero_grad()
            del data, bag_label, label

        # VALIDATION PHASE
        result, _ = evaluate(model, validation_dl)  # returns a results dict for metrics
        result["train_loss"] = sum(train_losses) / len(
            train_losses
        )  # torch.stack(train_losses).mean().item()
        result["train_acc"] = sum(train_acc) / len(train_acc)
        history.append(result)

        print(
            "Epoch [{}] : Train Loss {:.4f}, Train Acc {:.4f}, Val Loss {:.4f}, Val Acc {:.4f}".format(
                epoch,
                result["train_loss"],
                result["train_acc"],
                result["val_loss"],
                result["val_acc"],
            )
        )
        # Save best model / checkpointing stuff

        is_best = bool(result["val_acc"] >= best_acc)
        best_acc = max(result["val_acc"], best_acc)
        state = {
            "model_state_dict": model.state_dict(),
            "optim_state_dict": optim.state_dict(),
            "epoch": epoch,
            "loss": loss,
        }
        custom_model.save_checkpoint(state, is_best, model_savepath)

    return history


def k_fold_cross_val(
    X, y, og_labels, model, optim, metric_savepath, model_savepath, splits
):
    """k-fold cross validation for any number of RUNS where each run splits the data into the same amount of SPLITS."""
    KF = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)

    og_labels = np.asarray(og_labels)
    x = np.zeros(len(og_labels))
    # print(og_labels)
    # print(x)

    metric_savepath = metric_savepath
    model_savepath = model_savepath
    metric_savepath_og = metric_savepath
    model_savepath_og = model_savepath

    X = np.array(X)
    y = np.array(y)

    fold = 1
    for train_index, val_index in KF.split(x, og_labels):
        # Model
        model = choose_model(args.model)
        # model = model.to(device=device)
        loader_kwargs = {}
        if torch.cuda.is_available():
            #    #model.cuda()
            loader_kwargs = {"num_workers": 4, "pin_memory": True}

        # model.apply(init_he_weights) #not needed, as the default init scheme for Linear and Conv2d is already He init
        optim = choose_optimizer(args.optimizer, model)
        print("\nSuccessfully build and compiled the chosen model!")
        ###

        print("TRAIN:", train_index, "VAL:", val_index)
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # summarize stats
        train_0, train_1 = len(y_train[y_train == 0]), len(y_train[y_train == 1])
        val_0, val_1 = len(y_val[y_val == 0]), len(y_val[y_val == 1])
        print("Train: 0=%d, 1=%d, Val: 0=%d, 1=%d" % (train_0, train_1, val_0, val_1))

        print("Converting bags to batches")
        training_ds = preprocessing.convert_bag_to_batch(X_train, y_train)
        del X_train, y_train
        validation_ds = preprocessing.convert_bag_to_batch(X_val, y_val)
        del X_val, y_val

        # datasets which contain tuples of (list_data#, list_label#)
        print("Training DS contains {} patients".format(len(training_ds)))
        print("with labels: " + str([int(i[1][0]) for i in training_ds]))
        print(
            "Ratio of {} negative to {} positive cases.".format(
                len([int(i[1][0]) for i in training_ds if int(i[1][0]) == 0]),
                len([int(i[1][0]) for i in training_ds if int(i[1][0]) == 1]),
            )
        )
        print("\nValidation DS contains {} patients".format(len(validation_ds)))
        print("with labels: " + str([int(i[1][0]) for i in validation_ds]))
        print(
            "Ratio of {} negative to {} positive cases.".format(
                len([int(i[1][0]) for i in validation_ds if int(i[1][0]) == 0]),
                len([int(i[1][0]) for i in validation_ds if int(i[1][0]) == 1]),
            )
        )

        # random minority oversampling
        while len([row[1] for row in training_ds if row[1][0] == 0]) < len(
            [row[1] for row in training_ds if row[1][0] == 1]
        ):
            new_sample = random.choice([row for row in training_ds if row[1][0] == 0])
            training_ds.append(new_sample)

        print("\nModified Training DS after RANDOM MINORITY OVERSAMPLING")
        print("Training DS contains {} patients".format(len(training_ds)))
        print("with labels: " + str([int(i[1][0]) for i in training_ds]))
        print(
            "Ratio of {} negative to {} positive cases.".format(
                len([int(i[1][0]) for i in training_ds if int(i[1][0]) == 0]),
                len([int(i[1][0]) for i in training_ds if int(i[1][0]) == 1]),
            )
        )

        # perform train/val run

        # Data Generators
        train_dl = torch.utils.data.DataLoader(
            training_ds, batch_size=1, shuffle=True, **loader_kwargs
        )
        validation_dl = torch.utils.data.DataLoader(
            validation_ds, batch_size=1, shuffle=False, **loader_kwargs
        )

        # Training (n runs with k-fold cross validation)
        print("FOLD Nr. " + str(fold))
        print("Start of training for " + str(args.epochs) + " epochs.")
        history = fit(model, optim, train_dl, validation_dl, model_savepath)

        print("Plotting and saving loss and acc plots...")
        mil_metrics.plot_losses(history, metric_savepath)
        mil_metrics.plot_accuracy(history, metric_savepath)

        # Get best saved model from this run
        model, optim, _, _ = custom_model.load_checkpoint(model_savepath, model, optim)

        print("Computing and plotting confusion matrix...")
        y_hats, y_pred, y_true = get_predictions(model, validation_dl)
        mil_metrics.plot_conf_matrix(
            y_true,
            y_pred,
            metric_savepath,
            target_names=["Inflammation", "Cancer"],
            normalize=False,
        )

        print("Computing and plotting binary ROC-Curve")
        fpr, tpr, _ = mil_metrics.binary_roc_curve(y_true, y_hats)
        mil_metrics.plot_binary_roc_curve(fpr, tpr, metric_savepath)
        print("y_true:")
        print(y_true)
        print("y_hats:")
        print(y_hats)
        fpr, tpr, _ = mil_metrics.binary_roc_curve(y_true, y_hats)
        print("fpr:")
        print(fpr)
        print("tpr:")
        print(tpr)
        print("ROC AUC:")
        roc_auc = auc(fpr, tpr)
        print(roc_auc)

        ###################################

        # Hard Negative Mining
        false_positive_bags, attention_weights_list = mining.get_false_postive_bags(
            model, train_dl
        )
        print("-------------------------------------------------")
        print("Hard Negative Mining...")
        hard_negative_instances = mining.determine_hard_negative_instances(
            false_positive_bags, attention_weights_list
        )
        if not len(hard_negative_instances):
            print("No hard negative instances found!")
        else:
            new_bags = mining.new_bag_generation(
                hard_negative_instances, training_ds, n_clusters=4
            )
            training_ds = mining.add_back_to_dataset(training_ds, new_bags)

        # second training after HNM

        train_dl = torch.utils.data.DataLoader(
            training_ds, batch_size=1, shuffle=True, **loader_kwargs
        )
        validation_dl = torch.utils.data.DataLoader(
            validation_ds, batch_size=1, shuffle=False, **loader_kwargs
        )

        metric_savepath = metric_savepath + str(
            "_afterHNM"
        )  # modify savepath to reflect result changes after HNM
        model_savepath = model_savepath + str(
            "_afterHNM"
        )  # modify savepath to reflect result changes after HNM

        print("Start of SECOND training for " + str(args.epochs) + " epochs.")
        history = fit(model, optim, train_dl, validation_dl, model_savepath)
        print("Plotting and saving loss and acc plots for SECOND training after HNM...")
        mil_metrics.plot_losses(history, metric_savepath)
        mil_metrics.plot_accuracy(history, metric_savepath)

        # Get best saved model from this run
        model, optim, _, _ = custom_model.load_checkpoint(model_savepath, model, optim)

        print("Computing and plotting confusion matrix...")
        y_hats, y_pred, y_true = get_predictions(model, validation_dl)
        mil_metrics.plot_conf_matrix(
            y_true,
            y_pred,
            metric_savepath,
            target_names=["cancerous", "inflammation"],
            normalize=False,
        )

        print("Computing and plotting binary ROC-Curve")
        print("y_true:")
        print(y_true)
        print("y_hats:")
        print(y_hats)
        fpr, tpr, _ = mil_metrics.binary_roc_curve(y_true, y_hats)
        print("fpr:")
        print(fpr)
        print("tpr:")
        print(tpr)
        print("ROC AUC:")
        roc_auc = auc(fpr, tpr)
        print(roc_auc)

        mil_metrics.plot_binary_roc_curve(fpr, tpr, metric_savepath)

        fold += 1

        metric_savepath = metric_savepath_og
        model_savepath = model_savepath_og


# UTILS
#######
def get_device(gpu_switch=True):
    """Pick GPU if available, else run on CPU.
    Returns the corresponding device.
    """
    if torch.cuda.is_available() and gpu_switch:
        print("Running on GPU!\n")
        return torch.device("cuda")
    else:
        print("NOT running on GPU! Maybe a CUDA/Driver problem?\n")
        return torch.device("cpu")


def check_data(data):
    """Prints characteristic image/array values that are often
    useful to checkout.
    """
    print("\n")
    print("Data Check:")
    print("-----------")
    print("dtype: {}".format(data.dtype))
    print("max: {}".format(data.max()))
    print("min: {}".format(data.min()))
    print("mean: {}".format(np.round(data.mean(), decimals=5)))
    print("std: {}".format(np.round(data.std(), decimals=5)))
    print("\n")


#############
#############

# MAIN
######

if __name__ == "__main__":

    # Init
    global device
    device = get_device()
    print(device)

    global input_dim  # as called in choose_model
    input_dim = (
        150,
        150,
    )  # size of a single tile and also network input dimensionality

    metric_savepath = str(os.getcwd()) + "/path/to/save/to/"  # CHANGE THIS LINE!
    model_savepath = str(os.getcwd()) + "/path/to/save/to/2"  # CHANGE THIS LINE!

    args = parse_args()
    args_dict = vars(args)

    print("Called with args:")
    print(args)

    #############

    # Data
    print(
        "Loading data ..."
    )  # in this case there were 2 datasets, one for normal and one for cancerous samples
    datapath = "your/data/path"  # CHANGE THIS LINE!
    data = np.load(datapath, allow_pickle=True)
    X = data["X"]
    y = data["y"]

    del data, datapath

    datapath = "your/data/path/2"  # CHANGE THIS LINE!
    data = np.load(datapath, allow_pickle=True)
    X_1 = data["X"]
    y_1 = data["y"]

    X = np.concatenate((X, X_1))
    y = np.concatenate((y, y_1))

    del data, datapath, X_1, y_1

    print("Finished loading.\n")

    # Preprocess
    print("Preprocessing data into bags and labels...")
    X, y, og_labels = preprocess_data(X, y, input_dim)

    X = [np.einsum("bchw->bhwc", img) for img in X]
    X = [img / 255.0 for img in X]
    X = [preprocessing.zscore(img, axis=(0, 1, 2)) for img in X]
    X = [np.einsum("bhwc->bchw", img) for img in X]

    print("Shapes after preprocessing: ")
    print("X: " + str(len(X)))
    print("y: " + str(len(y)))
    print("Data shape of first sample: " + str(X[0].shape))
    print("Label shape of first sample: " + str(y[0].shape))

    check_data(X[0])

    #############

    # Model
    model = choose_model(args.model)
    # model = model.to(device=device)
    loader_kwargs = {}
    if torch.cuda.is_available():
        #    #model.cuda()
        loader_kwargs = {"num_workers": 4, "pin_memory": True}

    # model.apply(init_he_weights) #not needed, as the default init scheme for Linear and Conv2d is already He init
    optim = choose_optimizer(args.optimizer, model)
    print("\nSuccessfully build and compiled the chosen model!")
    #############

    ###

    k_fold_cross_val(
        X, y, og_labels, model, optim, metric_savepath, model_savepath, splits=3
    )

    print("Training finished. Now Testing...")

    # get best model after all runs
    # model, optim, _, _ = custom_model.load_checkpoint(model_savepath, model, optim)
    # test(#TODO) # CHANGE THIS LINE if you want to apply the model to external testing data


# END OF FILE
#############
