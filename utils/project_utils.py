import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os, time, torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import transforms


def get_clip_bounds(mean, std, dim):
    """
    Function computes the minimum and maximum clamping bounds
    using the mean and standard deviation of the dataset 
    
    Arguments:
        mean : Mean for each channel of the dataset
        std  : Standard deviation for each channel of the dataset
        dim  : Dimension of the input images of the model
    
    Returns:
        clip_min : Tensor with the minimum values allowed for each channel
        clip_max : Tensor with the maximum values allowed for each channel
    """
    
    clip_shape = [1, dim, dim]
    
    if isinstance(mean, (list, tuple, np.ndarray)):
        clip_min = []
        clip_max = []
        
        for i in range(len(mean)):
            clip_min.append(torch.full(clip_shape, (0.0 - mean[i]) / std[i]))
            clip_max.append(torch.full(clip_shape, (1.0 - mean[i]) / std[i]))
        
        clip_min = torch.cat(clip_min)
        clip_max = torch.cat(clip_max)
    else:
        clip_min = torch.full(clip_shape, (0.0 - mean) / std)
        clip_max = torch.full(clip_shape, (1.0 - mean) / std)
    
    return clip_min, clip_max


def deepfool(model, clip_min, clip_max, images, labels=None, l2_norm=True,
             batch_size=10, num_classes=10, overshoot=0.02, max_iters=50):
    """
    Function implements the DeepFool adversarial attack from the paper
    'DeepFool: a simple and accurate method to fool deep neural networks'
    [https://arxiv.org/pdf/1511.04599.pdf]
    
    Input Arguments:
        model       : Model to attack using DeepFool
        clip_min    : Minimum boundary of the adversarial images
        clip_max    : Maximum boundary of the adversarial images
        images      : Original images of the dataset
        labels      : True labels of the images of the dataset, else if set
                      to None, predictions of the model are used instead of
                      the true labels
                      (Default = None)
        l2_norm     : Flag used to determine if perturbations use L2 norm,
                      else if set to false, L-inf norm is used instead
                      (Default = True)
        batch_size  : Batch size used for the output logits of the model
                      (Default = 10)
        num_classes : Number of classes to use for the prediction result of
                      the aversarial images
                      (Default = 10)
        overshoot   : Constant used to scale the adversarial perturbations
                      (Default = 0.02)
        max_iters   : Maximum iterations allowed for generating an adversarial
                      image for each image
                      (Default = 50)
    
    Outputs:
        images_adv  : Adversarial images generated from DeepFool
        images_pert : Perturbations generated from DeepFool
        confs_adv   : Model confidences of the adversarial images
        labels_adv  : Model predictions of the adversarial images
        iters       : Total iterations it took for each adversarial image
    """
    
    images_adv = []
    images_pert = []
    confs_adv = []
    labels_adv = []
    iters = []
    
    device = images.get_device()
    
    # Create a tensor with all class labels
    # and split the tensor into batches
    k_all = torch.arange(0, num_classes).long().to(device)
    k_batch = k_all.split(batch_size)
    
    # Get the batch length and the size of the last batch,
    # in case it has a different size from the rest
    batch_length = len(k_batch)
    batch_size_last = len(k_batch[-1])
    
    # Use the negative log likelihood loss function without the
    # following softmax activation, as recommended in the paper
    loss_fn = nn.NLLLoss(reduction='sum')
    
    # Generate the adversarial image for each input image
    for i in range(len(images)):
        # Use a batch size of 1 for the input image and
        # intialize the perturbation with values of zero
        image = images[i].unsqueeze(dim=0).detach()
        r_hat = torch.zeros(image.shape).to(device)
        x = image
        
        # Duplicate the image so it has the same
        # batch size as the class label batch size
        x_batch = x.repeat(batch_size, 1, 1, 1)
        x_batch.requires_grad = True
        x_batch_last = x.repeat(batch_size_last, 1, 1, 1)
        x_batch_last.requires_grad = True
        
        # Get the model logit values of the image and
        # get the max confidence and predicted label
        f_k = model(x_batch)
        conf_i, k_i = f_k[0].max(dim=0)
        conf_i = conf_i.detach()
        
        # Create a list with labels not equal to the predicted label if
        # the true label is not known, or the true label if its known
        if labels is None:
            k_0 = k_i
        else:
            k_0 = labels[i]
        k_wrong = k_all[k_all != k_0]
        
        # Set the flag as true if the label 'k_0' is outside the tensor
        # containing the class indices with the range of 'num_classes'
        out_of_bounds = (False if (k_0 in k_all) else True)
        
        # Loop until max iterations are reached or if the
        # adversarial image causes the model to misclassify
        for j in range(max_iters):
            if not k_i.eq(k_0):
                break
            
            # Compute the gradient for the logit value of the
            # label 'k_0' if its out of bounds of 'num_classes'
            w_k0 = None
            if out_of_bounds:
                f_k[0][k_0].backward(retain_graph=True)
                w_k0 = x_batch.grad.detach().clone()[0].unsqueeze(dim=0)
                x_batch.grad.zero_()
            
            # Compute the gradients for all logit values of the model output
            w_k = []
            for k in range(batch_length):
                if k == (batch_length - 1):
                    f_k = model(x_batch_last)
                    x_batch = x_batch_last
                    loss_fn(f_k, k_batch[k]).backward(retain_graph=False)
                else:
                    loss_fn(f_k, k_batch[k]).backward(retain_graph=True)
                w_k.append(-x_batch.grad.detach().clone())
                x_batch.grad.zero_()
            w_k = torch.cat(w_k)
            
            # Compute the difference between gradient/logit values for the
            # wrong labels and the gradient/logit value for the label 'k_0'
            if out_of_bounds:
                w_p = w_k[k_wrong] - w_k0
            else:
                w_p = w_k[k_wrong] - w_k[k_0]
            f_p = (f_k[0][k_wrong] - f_k[0][k_0]).detach()
            
            # Compute the perturbation scalars and get the index for the
            # perturbation scalar that resulted in the smallest value
            if l2_norm:
                perts = f_p.abs() / w_p.flatten(start_dim=1).norm(dim=1, p=2)
            else:
                perts = f_p.abs() / w_p.flatten(start_dim=1).norm(dim=1, p=1)
            l_hat = perts.min(dim=0)[1]
            
            # Compute the minimum perturbation and sum it with the previous
            # perturbations (Authors of paper did not mention a constant
            # added to the perturbation scalar, but it helps reduce the
            # number of iterations needed to fool the classifier)
            if l2_norm:
                r_i = (perts[l_hat] + 1e-4) * w_p[l_hat] / w_p[l_hat].norm(p=2)
            else:
                r_i = (perts[l_hat] + 1e-4) * w_p[l_hat].sign()
            r_hat += r_i
            
            # Project the perturbation scaled with the overshoot
            # value over the input image and clamp the adversarial
            # image using the minimum and maximum bounds
            x = image + (1 + overshoot) * r_hat
            x = torch.max(torch.min(x, clip_max), clip_min)
            
            # Duplicate the adversarial image so it has the
            # same batch size as the class label batch size
            x_batch = x.repeat(batch_size, 1, 1, 1)
            x_batch.requires_grad = True
            x_batch_last = x.repeat(batch_size_last, 1, 1, 1)
            x_batch_last.requires_grad = True
            
            # Get the model logit values of the aversarial image
            # and get the max confidence and predicted label
            f_k = model(x_batch)
            conf_i, k_i = f_k[0].max(dim=0)
            conf_i = conf_i.detach()
            
            del w_k0, w_k, w_p, f_p
            del perts, l_hat, r_i
        
        images_adv.append(x)
        images_pert.append(x - image)
        confs_adv.append(conf_i.unsqueeze(dim=0))
        labels_adv.append(k_i.unsqueeze(dim=0))
        iters.append(j)
        
        del image, r_hat, x, x_batch, x_batch_last
        del f_k, conf_i, k_i, k_0, k_wrong
    
    images_adv = torch.cat(images_adv)
    images_pert = torch.cat(images_pert)
    confs_adv = torch.cat(confs_adv)
    labels_adv = torch.cat(labels_adv)
    
    del k_all, k_batch
    
    return images_adv, images_pert, confs_adv, labels_adv, iters


def fgsm(model, clip_min, clip_max, images, labels, eps=0.007):
    """
    Function implements the FGSM adversarial attack from the paper
    'Explaining and Harnessing Adversarial Examples'
    [https://arxiv.org/pdf/1412.6572.pdf]
    
    Input Variables:
        model    : Model to attack using FGSM
        clip_min : Minimum boundary of the adversarial images
        clip_max : Maximum boundary of the adversarial images
        images   : Original images of the dataset
        labels   : True labels of the images of the dataset
        eps      : Scalar used for the adversarial perturbations
                   (Default = 0.007)
    
    Output Variables:
        images_adv  : Adversarial images generated from FGSM
        images_pert : Perturbations generated from FGSM
        confs_adv   : Model confidences of the adversarial images
        labels_adv  : Model predictions of the adversarial images
    """
    
    images_grad = images.detach()
    images_grad.requires_grad = True
    
    loss_fn = nn.CrossEntropyLoss()
    
    # Compute the loss of the model outputs and perform
    # backpropagation to compute the gradients of the images
    outputs = model(images_grad)
    loss_fn(outputs, labels).backward()
    
    # Project the scaled gradient over the input images and clamp
    # the adversarial images using the minimum and maximum bounds
    images_adv = images.detach() + eps * images_grad.grad.sign()
    images_adv = torch.max(torch.min(images_adv, clip_max), clip_min)
    images_pert = images_adv - images.detach()
    
    # Get the confidences and predictions of the adversarial images
    with torch.no_grad():
        outputs = model(images_adv)
    confs_adv, labels_adv = outputs.max(dim=1)
    
    del images_grad, outputs
    
    return images_adv, images_pert, confs_adv, labels_adv


def compute_robustness(x, r_hat):
    """
    Function computes the adversarial robustness using the equation from the
    paper 'DeepFool: a simple and accurate method to fool deep neural networks'
    [https://arxiv.org/pdf/1511.04599.pdf]
    
    Input Variables:
        x     : Original images of the dataset
        r_hat : Perturbations generated from the images
    
    Output Variables:
        p_adv : Robustness value of each perturbation
    """
    
    p_adv = (r_hat.flatten(start_dim=1).norm(dim=1) \
        / x.flatten(start_dim=1).norm(dim=1)).cpu().numpy()
    
    return p_adv


def evaluate_attack(file_name, file_dir, device, model, dataset_loader,
                    clip_min, clip_max, adv_args, is_fgsm=True,
                    has_labels=False, l2_norm=True, verbose=True):
    """
    Function evaluates the adversarial attack method chosen (FGSM or DeepFool)
    and saves the results in a .csv file
    
    Input Variables:
        file_name      : Name of the .csv file
        file_dir       : Directory where the .csv file will be saved
        device         : Device to forward the tensors to (CPU or GPU)
        model          : Model to attack using FGSM or DeepFool
        dataset_loader : Dataset to use for the adversarial images
        clip_min       : Minimum boundary of the adversarial images
        clip_max       : Maximum boundary of the adversarial images
        adv_args       : Epsilon parameter for FGSM or parameters for DeepFool
                         eps or [batch_size, num_classes, overshoot, max_iters]
        is_fgsm        : Flag used to set the adversarial attack as FGSM, or
                         DeepFool if false
                         (Default = True)
        has_labels     : Flag used to determine if the DeepFool adversarial
                         attack has access to the true labels of the dataset
                         (Default = False)
        l2_norm        : Flag used to determine if DeepFool perturbations use L2
                         norm, else if set to false, L-inf norm is used instead
                         (Default = True)
        verbose        : Prints the batch progress if set to True
                         (Default = True)
    
    Output Variables:
        None
    """
    
    model.eval()
    
    file_path = os.path.join(file_dir, file_name)
    
    if not os.path.isdir(file_dir):
        os.makedirs(file_dir, exist_ok=True)
    
    if os.path.isfile(file_path):
        df = pd.read_csv(file_path)
    else:
        df = pd.DataFrame(columns=['batch_idx', 'correct', 'p_adv', 'time'])
        df.to_csv(file_path, index=False, header=True)
    
    if is_fgsm:
        name = 'FGSM'
    else:
        name = 'DeepFool'
    
    length_batches = len(dataset_loader)
    length_dataset = len(dataset_loader.dataset)
    length_df = len(df.index)
    
    if length_df < length_batches:
        for batch_idx, (images, labels) in enumerate(dataset_loader):
            if batch_idx < length_df:
                continue
            
            images, labels = images.to(device), labels.to(device)
            time_start = time.time()
            
            if is_fgsm:
                _, perts, _, labels_adv = fgsm(
                    model, clip_min, clip_max, images, labels, adv_args)
            elif has_labels:
                _, perts, _, labels_adv, _ = deepfool(
                    model, clip_min, clip_max, images, labels, l2_norm,
                    adv_args[0], adv_args[1], adv_args[2], adv_args[3])
            else:
                _, perts, _, labels_adv, _ = deepfool(
                    model, clip_min, clip_max, images, None, l2_norm,
                    adv_args[0], adv_args[1], adv_args[2], adv_args[3])
            
            time_batch = time.time() - time_start
            correct = (labels_adv == labels).sum().item()
            p_adv = np.mean(compute_robustness(images, perts))
            
            df = pd.DataFrame({'batch_idx' : [batch_idx],
                               'correct' : [correct],
                               'p_adv' : [p_adv],
                               'time' : [time_batch]})
            df.to_csv(file_path, mode='a', index=False, header=False)
            
            if verbose:
                if batch_idx < (length_batches - 1):
                    print('{:s} Batches Complete : ({:d} / {:d})'\
                        .format(name, batch_idx + 1, length_batches), end='\r')
                else:
                    print('{:s} Batches Complete : ({:d} / {:d})'\
                        .format(name, batch_idx + 1, length_batches))
            
            del images, labels, perts, labels_adv
    
    df = pd.read_csv(file_path).to_numpy().T
    
    test_error = (1.0 - (np.sum(df[1]) / length_dataset)) * 100.0
    robustness = np.mean(df[2])
    time_images = np.sum(df[3])
    time_image = time_images / length_dataset
    
    print('{:s} Test Error : {:.2f}%'.format(name, test_error))
    print('{:s} Robustness : {:.2e}'.format(name, robustness))
    print('{:s} Time (All Images) : {:.2f} s'.format(name, time_images))
    if int(time_image) != 0.0:
        print('{:s} Time (Per Image) : {:.2f} s'.format(name, time_image))
    elif int(time_image * 1e3) != 0.0:
        print('{:s} Time (Per Image) : {:.2f} ms'.format(name, time_image * 1e3))
    else:
        print('{:s} Time (Per Image) : {:.2f} us'.format(name, time_image * 1e6))


def display_attack(device, model, test_dataset, inv_tf, clip_min, clip_max,
                   fgsm_eps, deep_args, has_labels=False, l2_norm=True,
                   pert_scale=1.0, fig_rows=2, fig_width=25, fig_height=11,
                   label_map=None):
    """
    Function displays the images and predictions of the FGSM and DeepFool
    adversarial attacks
    
    Input Variables:
        device       : Device to forward the tensors to (CPU or GPU)
        model        : Model to attack using FGSM and DeepFool
        test_dataset : Test dataset used for the labels and images of the
                       adverarial attacks
        inv_tf       : Inverse transform to de-normalize the dataset images
        clip_min     : Minimum boundary of the adversarial images
        clip_max     : Maximum boundary of the adversarial images
        fgsm_eps     : Epsilon parameter for FGSM adversarial attack
        deep_args    : List of parameters for DeepFool adversarial attack
                       [batch_size, num_classes, overshoot, max_iters]
        has_labels   : Flag used to determine if the DeepFool adversarial
                       attack has access to the true labels of the dataset
                       (Default = False)
        l2_norm      : Flag used to determine if DeepFool perturbations use L2
                       norm, else if set to false, L-inf norm is used instead
                       (Default = True)
        pert_scale   : Scalar used to increase the perturbation visibility
                       (Default = 1.0)
        fig_rows     : Number of images to display in the figure
                       (Default = 2)
        fig_width    : Width of the figure
                       (Default = 25)
        fig_height   : Height of the figure
                       (Default = 11)
        label_map    : List that maps the label index to name of the label
                       (Default = None)
    
    Output Variables:
        None
    """
    
    model.eval()
    
    fig = plt.figure(figsize=(fig_width, fig_height), facecolor='white')
    
    test_loader = DataLoader(test_dataset, batch_size=fig_rows,
                             shuffle=True, num_workers=4)
    
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)
    
    if images.shape[1] == 1:
        is_gray = True
    else:
        is_gray = False
    
    with torch.no_grad():
        confs_pred, labels_pred = model(images).max(dim=1)
    
    advs_fgsm, perts_fgsm, confs_fgsm, labels_fgsm = fgsm(
        model, clip_min, clip_max, images, labels, fgsm_eps)
    
    if has_labels:
        advs_deep, perts_deep, confs_deep, labels_deep, iters_deep = deepfool(
            model, clip_min, clip_max, images, labels, l2_norm,
            deep_args[0], deep_args[1], deep_args[2], deep_args[3])
    else:
        advs_deep, perts_deep, confs_deep, labels_deep, iters_deep = deepfool(
            model, clip_min, clip_max, images, None, l2_norm,
            deep_args[0], deep_args[1], deep_args[2], deep_args[3])
    
    p_adv_fgsm = compute_robustness(images, perts_fgsm)
    p_adv_deep = compute_robustness(images, perts_deep)
    
    images_1 = inv_tf(images)
    images_2 = inv_tf(advs_fgsm)
    images_3 = inv_tf(advs_deep)
    images_4 = inv_tf(perts_fgsm * pert_scale)
    images_5 = inv_tf(perts_deep * pert_scale)
    
    for i in range(fig_rows):
        image_1 = transforms.ToPILImage()(images_1[i])
        image_2 = transforms.ToPILImage()(images_2[i])
        image_3 = transforms.ToPILImage()(images_3[i])
        image_4 = transforms.ToPILImage()(images_4[i])
        image_5 = transforms.ToPILImage()(images_5[i])
        
        label_true = labels[i].item()
        label_pred = labels_pred[i].item()
        label_fgsm = labels_fgsm[i].item()
        label_deep = labels_deep[i].item()
        
        if label_map is not None:
            label_true = label_map[label_true]
            label_pred = label_map[label_pred]
            label_fgsm = label_map[label_fgsm]
            label_deep = label_map[label_deep]
        
        fig.add_subplot(fig_rows, 5, (i * 5) + 1)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('true label : {:s}\npred label : {:s}\nconf score : {:.2f}'\
            .format(str(label_true), str(label_pred), confs_pred[i].item()),
            fontsize=16, loc='left')
        if i == 0:
            plt.title(label='Original', fontsize=20)
        if is_gray:
            plt.imshow(image_1, cmap='gray')
        else:
            plt.imshow(image_1)
        
        fig.add_subplot(fig_rows, 5, (i * 5) + 2)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('pred label : {:s}\nconf score : {:.2f}'\
            .format(str(label_fgsm), confs_fgsm[i].item()),
            fontsize=16, loc='left')
        if i == 0:
            plt.title(label='Adversarial (FGSM)', fontsize=20)
        if is_gray:
            plt.imshow(image_2, cmap='gray')
        else:
            plt.imshow(image_2)
        
        fig.add_subplot(fig_rows, 5, (i * 5) + 3)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('pred label : {:s}\nconf score : {:.2f}'\
            .format(str(label_deep), confs_deep[i].item()),
            fontsize=16, loc='left')
        if i == 0:
            plt.title(label='Adversarial (DeepFool)', fontsize=20)
        if is_gray:
            plt.imshow(image_3, cmap='gray')
        else:
            plt.imshow(image_3)
        
        fig.add_subplot(fig_rows, 5, (i * 5) + 4)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('robustness : {:.2e}\neps : {:s}'\
            .format(p_adv_fgsm[i], str(fgsm_eps)),
            fontsize=16, loc='left')
        if i == 0:
            plt.title(label='Perturbation (FGSM)', fontsize=20)
        if is_gray:
            plt.imshow(image_4, cmap='gray')
        else:
            plt.imshow(image_4)
        
        fig.add_subplot(fig_rows, 5, (i * 5) + 5)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('robustness : {:.2e}\novershoot : {:s}\niters : {:d}'\
            .format(p_adv_deep[i], str(deep_args[2]), iters_deep[i]),
            fontsize=16, loc='left')
        if i == 0:
            plt.title(label='Perturbation (DeepFool)', fontsize=20)
        if is_gray:
            plt.imshow(image_5, cmap='gray')
        else:
            plt.imshow(image_5)
    
    plt.show()
    
    del images, labels, confs_pred, labels_pred
    del advs_deep, perts_deep, confs_deep, labels_deep
    del advs_fgsm, perts_fgsm, confs_fgsm, labels_fgsm
    del images_1, images_2, images_3, images_4, images_5


def model_train(device, model, opt, epochs, loader_train, loader_val,
                print_step=1, clip_min=None, clip_max=None, adv_args=None,
                is_fgsm=True, has_labels=False, l2_norm=True, verbose=True):
    """
    Function is used to train the model with clean images of the dataset, or
    adversarial images of the dataset if adversarial parameters are passed to
    the function.
    
    Input Variables:
        device       : Device to forward the tensors to (CPU or GPU)
        model        : Model to train with the training set
        opt          : Optimizer used for training the model
        epochs       : Number of iterations the model is trained on the dataset
        loader_train : Dataset loader for the training set
        loader_val   : Dataset loader for the validation set
        print_step   : Epoch interval to print training and validation results
                       (Default = 1)
        clip_min     : Minimum boundary of the adversarial images
                       (Default = None)
        clip_max     : Maximum boundary of the adversarial images
                       (Default = None)
        adv_args     : Epsilon parameter for FGSM or parameters for DeepFool
                       eps or [batch_size, num_classes, overshoot, max_iters]
                       (Default = None)
        is_fgsm      : Flag used to set the adversarial method for training
                       as FGSM, or DeepFool if false
                       (Default = True)
        has_labels   : Flag used to determine if the DeepFool adversarial
                       attack has access to the true labels of the dataset
                       (Default = False)
        l2_norm      : Flag used to determine if DeepFool perturbations use L2
                       norm, else if set to false, L-inf norm is used instead
                       (Default = True)
        verbose      : Prints the training acc/loss per epoch if set to True
                       (Default = True)
    
    Output_Variables:
        train_accs   : List containing training accuracy for each epoch
        train_losses : List containing training loss for each epoch
        val_accs     : List containing validation accuracy for each epoch
        val_losses   : List containing validation loss for each epoch
    """
    
    if (clip_min is None) or (clip_min is None) or (adv_args is None):
        is_adv = False
    else:
        is_adv = True
    
    loss_fn = nn.CrossEntropyLoss()
    
    train_accs = []
    train_losses = []
    
    val_accs = []
    val_losses = []
    
    for epoch in range(epochs):      
        correct = 0
        losses = []
        
        model.train()
        
        for images, labels in loader_train:
            images, labels = images.to(device), labels.to(device)
            
            if is_adv:
                model.eval()
                if is_fgsm:
                    images_input, _, _, _ = fgsm(
                        model, clip_min, clip_max, images, labels, adv_args)
                elif has_labels:
                    images_input, _, _, _, _ = deepfool(
                        model, clip_min, clip_max, images, labels, l2_norm,
                        adv_args[0], adv_args[1], adv_args[2], adv_args[3])
                else:
                    images_input, _, _, _, _ = deepfool(
                        model, clip_min, clip_max, images, None, l2_norm,
                        adv_args[0], adv_args[1], adv_args[2], adv_args[3])
                model.train()
            else:
                images_input = images
            
            opt.zero_grad()
            
            outputs = model(images_input)
            loss = loss_fn(outputs, labels)
            
            correct += (outputs.max(dim=1)[1] == labels).sum().item()
            losses.append(loss.item())
            
            loss.backward()
            opt.step()
            
            del images, labels, images_input, outputs
        
        train_acc = correct / len(loader_train.dataset)
        train_loss = np.mean(losses)
        
        train_accs.append(train_acc)
        train_losses.append(train_loss)
        
        correct = 0
        losses = []
        
        model.eval()
        
        for images, labels in loader_val:
            images, labels = images.to(device), labels.to(device)
            
            if is_adv:
                if is_fgsm:
                    images_input, _, _, _ = fgsm(
                        model, clip_min, clip_max, images, labels, adv_args)
                elif has_labels:
                    images_input, _, _, _, _ = deepfool(
                        model, clip_min, clip_max, images, labels, l2_norm,
                        adv_args[0], adv_args[1], adv_args[2], adv_args[3])
                else:
                    images_input, _, _, _, _ = deepfool(
                        model, clip_min, clip_max, images, None, l2_norm,
                        adv_args[0], adv_args[1], adv_args[2], adv_args[3])
            else:
                images_input = images
            
            with torch.no_grad():
                outputs = model(images_input)
                loss = loss_fn(outputs, labels)
            
            correct += (outputs.max(dim=1)[1] == labels).sum().item()
            losses.append(loss.item())
            
            del images, labels, images_input, outputs
        
        val_acc = correct / len(loader_val.dataset)
        val_loss = np.mean(losses)
        
        val_accs.append(val_acc)
        val_losses.append(val_loss)
        
        if verbose and (epoch == 0 or (epoch + 1) % print_step == 0):
            print('Epoch [{:d}]'.format(epoch + 1))
            print('    Train Acc : {:.4f},  Train Loss : {:.4f}'\
                .format(train_acc, train_loss))
            print('      Val Acc : {:.4f},    Val Loss : {:.4f}'\
                .format(val_acc, val_loss))
    
    return train_accs, train_losses, val_accs, val_losses


def model_eval(device, model, loader_test, clip_min=None, clip_max=None,
               adv_args=None, is_fgsm=True, has_labels=False, l2_norm=True,
               verbose=True):
    """
    Function is used to evaluate the model with clean images of the dataset, or
    adversarial images of the dataset if adversarial parameters are passed to
    the function.
    
    Input Variables:
        device      : Device to forward the tensors to (CPU or GPU)
        model       : Model to evaluate with the test set
        loader_test : Dataset loader for the testing set
        clip_min    : Minimum boundary of the adversarial images
                      (Default = None)
        clip_max    : Maximum boundary of the adversarial images
                      (Default = None)
        adv_args    : Epsilon parameter for FGSM or parameters for DeepFool
                      eps or[batch_size, num_classes, overshoot, max_iters]
                      (Default = None)
        is_fgsm     : Flag used to set the adversarial method for evaluation
                      as FGSM, or DeepFool if false
                      (Default = True)
        has_labels  : Flag used to determine if the DeepFool adversarial
                      attack has access to the true labels of the dataset
                      (Default = False)
        l2_norm     : Flag used to determine if DeepFool perturbations use L2
                      norm, else if set to false, L-inf norm is used instead
                      (Default = True)
        verbose     : Prints the testing acc/loss if set to True
                      (Default = True)
    
    Output_Variables:
        test_acc  : Testing set accuracy of the model
        test_loss : Testing set loss of the model
    """
    
    if (clip_min is None) or (clip_min is None) or (adv_args is None):
        is_adv = False
    else:
        is_adv = True
    
    loss_fn = nn.CrossEntropyLoss()
    
    correct = 0
    losses = []
    
    model.eval()
    
    for images, labels in loader_test:
        images, labels = images.to(device), labels.to(device)
        
        if is_adv:
            if is_fgsm:
                images_input, _, _, _ = fgsm(
                    model, clip_min, clip_max, images, labels, adv_args)
            elif has_labels:
                images_input, _, _, _, _ = deepfool(
                    model, clip_min, clip_max, images, labels, l2_norm,
                    adv_args[0], adv_args[1], adv_args[2], adv_args[3])
            else:
                images_input, _, _, _, _ = deepfool(
                    model, clip_min, clip_max, images, None, l2_norm,
                    adv_args[0], adv_args[1], adv_args[2], adv_args[3])
        else:
            images_input = images
        
        with torch.no_grad():
            outputs = model(images_input)
            loss = loss_fn(outputs, labels)
        
        correct += (outputs.max(dim=1)[1] == labels).sum().item()
        losses.append(loss.item())
        
        del images, labels, images_input, outputs
    
    test_acc = correct / len(loader_test.dataset)
    test_loss = np.mean(losses)
    
    if verbose:
        if is_adv:
            if is_fgsm:
                print('Evaluation (FGSM Images)')
            else:
                print('Evaluation (DeepFool Images)')
        else:
            print('Evaluation (Clean Images)')
        print('     Test Acc : {:.4f},   Test Loss : {:.4f}'\
            .format(test_acc, test_loss))
    
    return test_acc, test_loss
