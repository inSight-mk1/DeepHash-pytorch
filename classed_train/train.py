import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import PIL

######################################################################
# Helper Functions
# ----------------
# 
# Before we write the code for adjusting the models, lets define a few
# helper functions.
# 
# Model Training and Validation Code
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# The ``train_model`` function handles the training and validation of a
# given model. As input, it takes a PyTorch model, a dictionary of
# dataloaders, a loss function, an optimizer, a specified number of epochs
# to train and validate for, and a boolean flag for when the model is an
# Inception model. The *is_inception* flag is used to accomodate the
# *Inception v3* model, as that architecture uses an auxiliary output and
# the overall model loss respects both the auxiliary output and the final
# output, as described
# `here <https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958>`__.
# The function trains for the specified number of epochs and after each
# epoch runs a full validation step. It also keeps track of the best
# performing model (in terms of validation accuracy), and at the end of
# training returns the best performing model. After each epoch, the
# training and validation accuracies are printed.
# 


def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25, is_inception=False, disp_every=10,
                batch_size=8):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    i = 0
    for epoch in range(num_epochs):
        i += 1
        scheduler.step()
        print("Learning Rate for Epoch {} is {} ".format(epoch + 1, scheduler.get_lr()))
        print('-' * 20)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            display_loss = 0.0
            display_corrects = 0

            # Iterate over data.
            iters = 0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                loss_ = loss.item() * inputs.size(0)
                correct_ = torch.sum(preds == labels.data)
                # print(preds, labels.data)
                # print('correct', correct_)
                running_loss += loss_
                display_loss += loss_
                running_corrects += correct_
                display_corrects += correct_

                if phase == 'train' and iters % disp_every == disp_every - 1:
                    display_loss /= (disp_every * batch_size)
                    # writer.add_scalars('training loss', {'total loss': running_loss},
                    #                    epoch * len(train_data_loader) + i)

                    time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + ' '
                    # print(display_corrects.double(), disp_every * batch_size)
                    display_acc = display_corrects.double() / (disp_every * batch_size)
                    # print(display_corrects, disp_every * batch_size)
                    print(time_str + 'Epoch: %03d Iter: %06d Total: %06d' % (epoch, iters, len(dataloaders[phase]))
                          + '   ' + 'learning rate: {}'.format(scheduler.get_lr()[0]))
                    print(' ' * 20 + 'loss: %.4f, acc: %.3f' % (display_loss, display_acc))
                    display_loss = 0.0
                    display_corrects = 0

                iters += 1

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()
        if i % save_every == 0:
            model_fname = 'weights_epoch%03d.pkl' % (i)
            model_path = os.path.join(save_path, model_fname)
            torch.save(model, model_path)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


# Notice, many of the models have similar output structures, but each must
# be handled slightly differently. Also, check out the printed model
# architecture of the reshaped network and make sure the number of output
# features is the same as the number of classes in the dataset.
# 

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    print(model_name)
    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet152":
        """ Resnet101
        """
        model_ft = models.resnet152(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet50":
        """ Resnet50
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3 
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


if __name__ == '__main__':
    print("PyTorch Version: ", torch.__version__)
    print("Torchvision Version: ", torchvision.__version__)

    # Top level data directory. Here we assume the format of the directory conforms
    #   to the ImageFolder structure
    data_dir = "D:\\dataset\\consecutive_vehicles_v1\\dataset"

    # Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
    model_name = "resnet152"
    save_dir = "0923_%s" % model_name
    save_path = os.path.join('pkls', save_dir)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Number of classes in the dataset
    num_classes = 190

    # Batch size for training (change depending on how much memory you have)
    batch_size = 20

    # Number of epochs to train for
    num_epochs = 30

    # Flag for feature extracting. When False, we finetune the whole model,
    #   when True we only update the reshaped layer params
    feature_extract = False

    # Training params
    base_lr = 0.001
    lr_decay_every = 10
    gamma = 0.5
    save_every = 1

    # Initialize the model for this run
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

    # Print the model we just instantiated
    print(model_ft)

    ######################################################################
    # Load Data
    # ---------
    #
    # Now that we know what the input size must be, we can initialize the data
    # transforms, image datasets, and the dataloaders. Notice, the models were
    # pretrained with the hard-coded normalization values, as described
    # `here <https://pytorch.org/docs/master/torchvision/models.html>`__.
    #

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            # transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            # transforms.Grayscale(num_output_channels=3),
            transforms.RandomRotation(15, PIL.Image.BICUBIC, expand=True),
            transforms.Resize([input_size, input_size]),
            transforms.ColorJitter(brightness=1.0, contrast=1.0),
            # transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            # transforms.Grayscale(num_output_channels=3),
            transforms.Resize([input_size, input_size]),
            # transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    print("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    print('train', train_dir)
    print('val', val_dir)
    train_dataset = datasets.ImageFolder(train_dir, data_transforms['train'])
    val_dataset = datasets.ImageFolder(val_dir, data_transforms['val'])

    print(train_dataset.class_to_idx)
    # total_len = len(total_dataset)
    # train_ratio = 0.5
    # val_ratio = 0.25
    # test_ratio = 0.25
    # train_len = int(train_ratio * total_len)
    # val_len = int(val_ratio * total_len)
    # test_len = total_len - train_len - val_len
    # split_len = [train_len, val_len, test_len]
    # subset = torch.utils.data.random_split(total_dataset, split_len)

    # image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    # Create training and validation dataloaders
    datasets_dict = {'train': train_dataset, 'val': val_dataset}
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(datasets_dict[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in
        ['train', 'val']}

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ######################################################################
    # Create the Optimizer

    # Send the model to GPU
    model_ft = model_ft.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t", name)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=base_lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=lr_decay_every, gamma=gamma)

    ######################################################################
    # Run Training and Validation Step
    # --------------------------------
    #
    # Finally, the last step is to setup the loss for the model, then run the
    # training and validation function for the set number of epochs. Notice,
    # depending on the number of epochs this step may take a while on a CPU.
    # Also, the default learning rate is not optimal for all of the models, so
    # to achieve maximum accuracy it would be necessary to tune for each model
    # separately.
    #

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, scheduler, num_epochs=num_epochs,
                                 is_inception=(model_name == "inception"), disp_every=20, batch_size=batch_size)

    model_fname = 'best_model.pkl'
    # model_ptname = 'best_model.pt'
    model_path = os.path.join(save_path, model_fname)
    # model_ptpath = os.path.join(save_path, model_ptname)
    torch.save(model_ft, model_path)
    # input = torch.randn((1, 3, 224, 224))
    # input = input.to(device)
    # traced_script_module = torch.jit.trace(model_ft, input)
    # traced_script_module.save(model_ptpath)

    ######################################################################
    # Comparison with Model Trained from Scratch
    # ------------------------------------------
    #
    # Just for fun, lets see how the model learns if we do not use transfer
    # learning. The performance of finetuning vs. feature extracting depends
    # largely on the dataset but in general both transfer learning methods
    # produce favorable results in terms of training time and overall accuracy
    # versus a model trained from scratch.
    #

    # Initialize the non-pretrained version of the model used for this run
    # scratch_model, _ = initialize_model(model_name, num_classes, feature_extract=False, use_pretrained=False)
    # scratch_model = scratch_model.to(device)
    # scratch_optimizer = optim.SGD(scratch_model.parameters(), lr=0.001, momentum=0.9)
    # scratch_criterion = nn.CrossEntropyLoss()
    # _, scratch_hist = train_model(scratch_model, dataloaders_dict, scratch_criterion, scratch_optimizer,
    #                               num_epochs=num_epochs, is_inception=(model_name == "inception"))

    # Plot the training curves of validation accuracy vs. number
    #  of training epochs for the transfer learning method and
    #  the model trained from scratch
    ohist = []
    shist = []

    ohist = [h.cpu().numpy() for h in hist]
    # shist = [h.cpu().numpy() for h in scratch_hist]

    plt.title("Validation Accuracy vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Validation Accuracy")
    plt.plot(range(1, num_epochs + 1), ohist, label="Pretrained")
    # plt.plot(range(1, num_epochs + 1), shist, label="Scratch")
    plt.ylim((0, 1.))
    plt.xticks(np.arange(1, num_epochs + 1, 1.0))
    plt.legend()
    plt.show()
