# -*- coding: utf-8 -*-
import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '5'
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torch.nn.functional as F
import Pretrained_FR_Models.irse as irse
import Pretrained_FR_Models.facenet as facenet
import Pretrained_FR_Models.ir152 as ir152
from utils import *

output_dir_list = ["./mobile_face/","./irse50/"]
model_list = [["irse50","facenet","ir152"],["ir152","facenet","mobile_face"]]


def cos_simi(emb_before_pasted, emb_target_img):
    """
    :param emb_before_pasted: feature embedding for the generated adv-makeup face images
    :param emb_target_img: feature embedding for the victim target image
    :return: cosine similarity between two face embeddings
    """
    return torch.mean(torch.sum(torch.mul(emb_target_img, emb_before_pasted), dim=1)
                      / emb_target_img.norm(dim=1) / emb_before_pasted.norm(dim=1))


def cal_target_loss(before_pasted, target_img, model_name):
    fr_model = ir152.IR_152((112, 112))
    fr_model.load_state_dict(torch.load('./Pretrained_FR_Models/ir152.pth'))
    input_size = (112, 112)
    if model_name == 'ir152':
        input_size = (112, 112)
        # self.models_info[model_name][0].append((112, 112))
        fr_model = ir152.IR_152((112, 112))
        fr_model.load_state_dict(torch.load('./Pretrained_FR_Models/ir152.pth'))
    if model_name == 'irse50':
        input_size = (112, 112)
        fr_model = irse.Backbone(50, 0.6, 'ir_se')
        fr_model.load_state_dict(torch.load('./Pretrained_FR_Models/irse50.pth'))
    if model_name == 'mobile_face':
        input_size = (112, 112)
        fr_model = irse.MobileFaceNet(512)
        fr_model.load_state_dict(torch.load('./Pretrained_FR_Models/mobile_face.pth'))
    if model_name == 'facenet':
        input_size = (160, 160)
        fr_model = facenet.InceptionResnetV1(num_classes=8631, device=device)
        fr_model.load_state_dict(torch.load('./Pretrained_FR_Models/facenet.pth'))
    fr_model.to(device)
    fr_model.eval()

    before_pasted_resize = F.interpolate(before_pasted, size=input_size, mode='bilinear')
    target_img_resize = F.interpolate(target_img, size=input_size, mode='bilinear')

    # Inference to get face embeddings
    emb_before_pasted = fr_model(before_pasted_resize)
    emb_target_img = fr_model(target_img_resize).detach()

    # Cosine loss computing
    cos_loss =  cos_simi(emb_before_pasted, emb_target_img)
    return cos_loss

def FGSM_Attack(img_before, target, model_names, eps=0.025):
    img_adv = img_before.clone().detach().requires_grad_(True)

    all_losses = []
    for model_name in model_names:
        cos_loss = cal_target_loss(img_adv, target, model_name)
        all_losses.append(cos_loss)
    total_loss = torch.mean(torch.stack(all_losses))

    total_loss.backward()

    with torch.no_grad():
        perturbation = eps * img_adv.grad.sign()
        img_adv = img_before + perturbation

        img_adv = torch.clamp(img_adv, min=0, max=1)


    print(total_loss.item())  

    return img_adv


avg_ssim = 0.0
avg_psnr = 0.0
if __name__ == "__main__":
    path = './before_aligned_600'
    target_img = cv2.imread("./target_aligend_600/Camilla_Parker_Bowles_0002.jpg") / 255.0
    target_img = torch.from_numpy(target_img).permute(2, 0, 1).to(torch.float32).to(device).unsqueeze(0)
    ssim_list = []
    psnr_list = []

    i = 0
    for root, dirs, files in os.walk(path, topdown=True):
        for index in range(2):
            output_dir = output_dir_list[index]
            print("Black Attack:", output_dir.split("/")[1])
            m = model_list[index]
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            for name in files:
                file_path = os.path.join(root, name)
                img = cv2.imread(file_path) / 255.0
                print(name)
                img = torch.from_numpy(img).permute(2, 0, 1).to(torch.float32).to(device).unsqueeze(0)

                attacked_img =FGSM_Attack(img, target_img, m)
                sim = calculate_ssim(img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255.0,
                                     attacked_img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255.0)
                psr = calculate_psnr(img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255.0,
                                     attacked_img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255.0)
                avg_ssim = avg_ssim + sim
                avg_psnr = avg_psnr + psr
                img_np = attacked_img.squeeze(0).permute(1, 2, 0).cpu().detach().numpy() * 255.0
                cv2.imwrite(output_dir + str(i) + '.png', img_np)
                i = i + 1

    print("SSIM:", avg_ssim / i)
    print("PSRN:", avg_psnr / i)
          
