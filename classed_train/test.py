from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
# import matplotlib.pyplot as plt
import time
import os
import copy
import PIL
import cv2


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract=False, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
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


def predict_multi(model, dataloaders, criterion, dst_path=None):
    since = time.time()

    dataloader = dataloaders['test']
    model.eval()  # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0

    # Iterate over data.
    i = 0
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        # print('in', inputs)
        labels = labels.to(device)
        # print('lb', labels)

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

        # print(preds, preds == labels.data)

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        # if preds != labels.data:
        img = inputs[0].permute(1, 2, 0).cpu().numpy()
        # print("input", img.shape)
        # print("label", labels)
        result = preds[0].item()
        result_str = '%02d' % result
        filename = '%06d.jpg' % i
        if dst_path is not None:
            save_folder = os.path.join(dst_path, result_str)
            if not os.path.exists(save_folder):
                os.mkdir(save_folder)
            save_path = os.path.join(save_folder, filename)
            img = img * 255
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(save_path, img)
            print(filename)
        else:
            print("pred", preds[0].item())
            cv2.imshow("img", img)
            cv2.waitKey(0)

        i += 1

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)

    print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))


def patch_detection(model, dataloaders, criterion):
    dataloader = dataloaders['test']
    model.eval()  # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0

    # Iterate over data.
    for inputs, labels in dataloader:
        inputs = inputs.to(device)

        patch_size = 224
        # print(inputs.shape)
        patch_row_cnt = inputs.shape[2] // patch_size
        patch_col_cnt = inputs.shape[3] // patch_size
        patch_cnt = patch_row_cnt * patch_col_cnt
        patch_inputs = torch.zeros((patch_cnt, 3, patch_size, patch_size)).to(device)
        # print(patch_row_cnt, patch_col_cnt, patch_inputs.shape)

        # generate patch label map
        x = 0
        y = 0
        k = 0
        # print(src_image.shape)
        # print(patch_row_cnt, patch_col_cnt)
        for i in range(patch_row_cnt):
            for j in range(patch_col_cnt):
                # print(y, y + patch_size, x, x + patch_size)
                patch_inputs[k] = inputs[0, :, y:y + patch_size, x:x + patch_size]
                k += 1
                x += patch_size

            y += patch_size
            x = 0

        # forward
        # track history if only in train

        with torch.set_grad_enabled(False):
            outputs = model(patch_inputs)
            _, preds = torch.max(outputs, 1)

        # print(preds)
        # for i in range(patch_cnt):
        #     disp_image = patch_inputs[i].permute(1, 2, 0).cpu().numpy()
        #     cv2.imshow('patch', disp_image)
        #     cv2.waitKey(0)

        inputs_np = inputs.cpu().detach().squeeze(dim=0).numpy()
        inputs_bytes = inputs_np * 255
        inputs_bytes = inputs_bytes.astype(np.uint8)

        preds_np = preds.cpu().detach().numpy()
        preds_mask = np.zeros_like(inputs_np, dtype=np.uint8)
        preds_img = preds_mask.transpose((1, 2, 0))
        inputs_img = inputs_bytes.transpose((1, 2, 0))

        inputs_img = cv2.cvtColor(inputs_img, cv2.COLOR_BGR2RGB)

        # generate patch label map
        x = 0
        y = 0
        k = 0
        # print(src_image.shape)
        # print(patch_row_cnt, patch_col_cnt)
        for i in range(patch_row_cnt):
            for j in range(patch_col_cnt):
                # print(y, y + patch_size, x, x + patch_size)
                one_mask_img = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
                if preds_np[k] == 0:
                    one_mask_img[:] = 255
                elif preds_np[k] == 1:  # crack - Coral
                    one_mask_img[..., 0] = 80
                    one_mask_img[..., 1] = 127
                    one_mask_img[..., 2] = 255
                elif preds_np[k] == 2:  # breakage - Crimson
                    one_mask_img[..., 0] = 60
                    one_mask_img[..., 1] = 20
                    one_mask_img[..., 2] = 200
                elif preds_np[k] == 3:  # corrosion - DarkSeaGreen
                    one_mask_img[..., 0] = 143
                    one_mask_img[..., 1] = 188
                    one_mask_img[..., 2] = 143
                preds_img[y:y + patch_size, x:x + patch_size, :] = one_mask_img
                k += 1
                x += patch_size

            y += patch_size
            x = 0


        disp_image = cv2.addWeighted(preds_img, 0.2, inputs_img, 0.8, 0)
        disp_image = cv2.resize(disp_image, (1920, 1080))
        cv2.imshow('disp', disp_image)
        cv2.waitKey(0)

        # print(preds, preds == labels.data)

        # statistics
    #     running_loss += loss.item() * inputs.size(0)
    #     running_corrects += torch.sum(preds == labels.data)
    #     if preds != labels.data:
    #         img = inputs[0].permute(1, 2, 0).cpu().numpy()
    #         # print("input", img.shape)
    #         print("label", labels)
    #         print("pred", preds)
    #         cv2.imshow("img", img)
    #         cv2.waitKey(0)
    #
    # epoch_loss = running_loss / len(dataloader.dataset)
    # epoch_acc = running_corrects.double() / len(dataloader.dataset)
    #
    # print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))


# def all_path(dirname):
#
#     result = []
#     fn_res = []
#
#     for maindir, subdir, file_name_list in os.walk(dirname):
#
#         # print("1:", maindir)
#         # print("2:", subdir)
#         # print("3:", file_name_list)
#
#         for filename in file_name_list:
#             apath = os.path.join(maindir, filename)
#             result.append(apath)
#             fn_res.append(filename)
#
#     return result, fn_res
#
# total_cnt = 0
# right_cnt = 0
# for i in range(1):
#     str_i = str(i)
#     if len(str_i) == 1:
#         str_i = '0' + str_i
#     test_dir = '...'
#     test_i_dir = os.path.join(test_dir, str_i)
#     res, fn_res = all_path(test_i_dir)
#     for fp in res:
#         pil_img = PIL.Image.open(fp)
#         pil_img = pil_img.resize((224, 224))
#         img = np.array(pil_img)
#         img = img / 255
#         t_img = torch.from_numpy(img)
#         t_img = t_img.unsqueeze(0)
#         t_img = t_img.permute((0, 3, 1, 2))
#         t_img = t_img.to(device, dtype=torch.float)
#         oup = net(t_img)
#         oup = oup.cpu().detach().numpy()
#         idx = np.argmax(oup)
#         if idx == i:
#             right_cnt += 1
#         total_cnt += 1
#         print(fp, idx, right_cnt, total_cnt)


if __name__ == '__main__':
    print("PyTorch Version: ", torch.__version__)
    print("Torchvision Version: ", torchvision.__version__)

    batch_size = 1
    num_classes = 5
    model_name = "densenet"
    load_dir = '0715_hangqian_dataset_densenet'
    load_path = os.path.join('pkls', load_dir)

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net, input_size = initialize_model(model_name, num_classes)

    data_transforms = {
        'test': transforms.Compose([
            # transforms.RandomResizedCrop(input_size),
            # transforms.RandomHorizontalFlip(),
            # transforms.Grayscale(num_output_channels=3),
            # transforms.RandomRotation(5, PIL.Image.BICUBIC, expand=True),
            transforms.Resize([input_size, input_size]),
            # transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    patch_transforms = {
        'test': transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    patch_test_path = "C:\\Users\\admin\\dongwei\\workspace\\dataset\\defeat_seg\\patch_test"
    cls_test_path = 'C:\\Users\\admin\\dongwei\\workspace\\dataset\\axles_cnt_cls\\dataset\\test'
    # predict_path = "G:\\Code\\dataset\\defeat_patch\\224_uav\\original_images"
    test_dataset = datasets.ImageFolder(cls_test_path, data_transforms['test'])  # Note: BOTH TWO params need to be checked
    datasets_dict = {'test': test_dataset}
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(datasets_dict[x], batch_size=batch_size, shuffle=False, num_workers=4) for x in
        ['test']}

    criterion = nn.CrossEntropyLoss()

    model_fname = 'best_model_with_jitter.pkl'
    model_path = os.path.join(load_path, model_fname)
    # net.load_state_dict(torch.load(model_path))2
    net = torch.load(model_path)
    print("Successfully loaded trained ckpt at {}".format(model_path))
    net.to(device)
    # patch_detection(net, dataloaders_dict, criterion)
    predict_multi(net, dataloaders_dict, criterion, dst_path='C:\\Users\\admin\\dongwei\\workspace\\dataset\\axles_cnt_cls\\test_results')
