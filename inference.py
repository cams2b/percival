import pandas as pd
import numpy as np
import SimpleITK as sitk
import torch
from train_operations.percival import percival


def single_inference_CT():
    test_img_path = ''
    print(test_img_path)
    img_weights = '/cbica/home/beechec/research/model_weights/foundation_percival/percival_checkpoint/weights/image_encoder_epoch_1_loss_1839.1098633.pth'
    in_channels = 1
    projection_dim = 512
    king_parsival = percival(
            name='king_parsival', 
            in_channels=in_channels, 
            projection_dim=projection_dim, 
            img_size=(128, 256, 256))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    king_parsival.to(device)
    king_parsival.load_image_encoder(path=img_weights)
    z_img = king_parsival.inference_from_path(img_path=test_img_path, device=device)


    res_df = king_parsival.diagnostic_inference_all_conditions(img_path=test_img_path, device=device)
    print(res_df)
    print(np.sum(res_df['predicted_label'].values))
    print('[INFO] DONE')

if __name__ == '__main__':
    single_inference_CT()
