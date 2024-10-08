"""
Evaluation of the model, and save the video colorized
from the example image passed
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from PIL import Image, ImageOps
import cv2
from torch.autograd import Variable
import torchvision.models as models

from architectures.ViT import *
from torchvision import io
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

import read_data as ld

from utils import *

import shutil
from architectures.color_model_simple import ColorNetwork
from architectures.flow import Flow_Network

dataLoader = ld.ReadData()

image_size = 128
device = "cuda"
# video_class = "parkour"
str_dt = "swin_unet_20230621_205446"
dataset = "DAVIS_test"
batch_size = 1
data_type = "test"

# List all classes to be evaluated
images_paths = f"./data/{data_type}/{dataset}"
img_classes = os.listdir(images_paths)

# torch tensor to image
def to_img(x):
    """
    Converts the input tensor `x` to an image tensor.

    Args:
        x (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: The converted image tensor.
    """
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, image_size, image_size)
    return x

# ================ Read Data ===================
root_model_path = "models"
# avaliable_models = os.listdir(root_model_path)
avaliable_models = [str_dt]

pbar = tqdm(avaliable_models)
# ================ Loop all videos inside gray folder =====================
for str_dt in pbar:
    pbar.set_description(f"Processing: {str_dt}")

    temp_path = f"temp/{dataset}/"

    os.makedirs(temp_path, exist_ok=True)

    # ================ Read Video =====================
    path_gray_video = f"./data/videos/{dataset}_gray/"
    try:
        list_gray_videos = os.listdir(path_gray_video)
    except FileNotFoundError:
        # Create the gray version of the videos
        create_gray_videos(dataset, path_gray_video)
        list_gray_videos = os.listdir(path_gray_video)

    # ================ Read Model =====================
    model_path = f'{root_model_path}/{str_dt}/color_network.pth'

    # try:
    model = load_trained_model(model_path, image_size, device)
    color_network = Vit_neck().to(device)
    swin_model = models.swin_v2_t(weights=models.Swin_V2_T_Weights.IMAGENET1K_V1).to("cuda").features

    # Define the pretrained Flow Network
    flow_model = Flow_Network().to("cuda")
    flow_model.eval()
    
    pbar = tqdm(list_gray_videos)
    # ================ Loop all videos inside gray folder =====================
    for video_name in pbar:
        pbar.set_description(f"Processing: {video_name}")

        vidcap = cv2.VideoCapture(f"{path_gray_video}{video_name}")
        success,image = vidcap.read()
        count = 0
        list_frames = []

        path_temp_gray_frames = f"{temp_path}{video_name.split('.')[0]}"
        if not os.path.exists(path_temp_gray_frames):

            os.makedirs(f"{path_temp_gray_frames}/images/", exist_ok=True)

            while success:
                cv2.imwrite(f"{path_temp_gray_frames}/images/{str(count).zfill(5)}.jpg", image)     # save frame as JPEG file      
                success,image = vidcap.read()
                list_frames.append(image)
                count += 1
        # print("Finsh")


    # ================ Read images to make the video =====================
        dataloader = dataLoader.create_dataLoader(path_temp_gray_frames, image_size, batch_size, shuffle=False)

        example_path = f"./data/train/{dataset}/{video_name}/"

        # example_img = Image.open(f"{example_path}{str(count-1).zfill(5)}.jpg", )
        example_img = Image.open(f"./data/{data_type}/{dataset}/{video_name.split('.')[0]}/00010.jpg")
        example_img = transforms.functional.pil_to_tensor(example_img.resize((image_size, image_size))).to(device)
        example_img = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(example_img.type(torch.float32))


    # ============== Frame Production ===================
        imgs_2 = []
        imgs_2_gray = []

        outs = []
        # path to save colored frames
        colored_frames_save = f"temp_result/{dataset}/{str_dt}/{video_name}/"

        if os.path.exists(colored_frames_save):
            shutil.rmtree(colored_frames_save)

        os.makedirs(colored_frames_save, exist_ok=True)

        # path so save videos
        colored_video_path = f"videos_output/{str_dt}/{video_name}/"
        os.makedirs(colored_video_path, exist_ok=True)


        with torch.no_grad():
            model.eval()
            imgs_data = []

            for idx, data in enumerate(dataloader):

                img, img_gray, img_color, next_frame = create_samples(data)

                img_color = img_color.to(device)
                img_gray = img_gray.to(device)
                next_frame = next_frame.to(device)

                imgs_data.append(img_gray)

                labels = color_network(img_color)

                flow = flow_model(img_gray, next_frame)

                # img_frame = transforms.Grayscale(num_output_channels=1)(img_frame)
                out = model(img_gray, labels, flow, swin_model)
                outs.append(out)

            for idx, frame in enumerate(outs):
                save_image(to_img(frame), f"{colored_frames_save}{str(idx).zfill(5)}.jpg")

        frame_2_video(colored_frames_save, f"{colored_video_path}/{video_name}_colored.mp4")

print("Evaluation Finish")
