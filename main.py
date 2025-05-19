import os
import glob
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral

class AttentionBlock(nn.Module):
    """
    Reused AttentionBlock defined in training.
    """

    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class AttentionUNet(nn.Module):
    """
    U-Net architecture with attention mechanism.
    """

    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(AttentionUNet, self).__init__()
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        for feature in features:
            self.encoder.append(self.double_conv(in_channels, feature))
            in_channels = feature

        self.bottleneck = self.double_conv(features[-1], features[-1]*2)

        self.up_transpose = nn.ModuleList()
        self.attention = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for feature in reversed(features):
            self.up_transpose.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.attention.append(
                AttentionBlock(F_g=feature, F_l=feature, F_int=feature // 2)
            )
            self.decoder.append(
                self.double_conv(feature*2, feature)
            )
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        """
        Forward pass for AttentionUNet.
        """
        skip_connections = []
        for encode in self.encoder:
            x = encode(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(len(self.up_transpose)):
            x = self.up_transpose[idx](x)
            skip_connection = skip_connections[idx]
            attn_skip = self.attention[idx](g=x, x=skip_connection)
            if x.shape != attn_skip.shape:
                x = nn.functional.interpolate(x, size=attn_skip.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([attn_skip, x], dim=1)
            x = self.decoder[idx](x)
        return self.final_conv(x)

    def double_conv(self, in_channels, out_channels):
        """
        Two consecutive convolutional layers with BatchNorm and ReLU.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

def dense_crf_postprocess(image, mask_prob):
    """
    Refine predicted mask using DenseCRF.

    @param image: RGB image (H, W, 3)
    @param mask_prob: Predicted probability map (H, W)
    @return: Refined probability map after DenseCRF
    """
    H, W = image.shape[:2]
    prob = np.stack([1 - mask_prob, mask_prob], axis=0)
    unary = unary_from_softmax(prob)
    unary = np.ascontiguousarray(unary)
    d = dcrf.DenseCRF2D(W, H, 2)
    d.setUnaryEnergy(unary)
    feats = create_pairwise_bilateral(sdims=(10, 10), schan=(13, 13, 13), img=image, chdim=2)
    d.addPairwiseEnergy(feats, compat=10)
    Q = d.inference(5)
    refined_prob = np.array(Q).reshape((2, H, W))[1]
    return refined_prob

def predict_single_image(image_bgr, model, device, threshold=0.5):
    """
    Predict mask and probability map for a single image.

    @param image_bgr: Input image in BGR format
    @param model: Trained model
    @param device: Computation device
    @param threshold: Threshold for binarizing probability map
    @return: Binary mask and raw probability
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    augmented = transform(image=image_rgb)
    input_tensor = augmented['image'].unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    output = torch.sigmoid(output)
    prob = output.squeeze().cpu().numpy()
    mask = (prob > threshold).astype(np.uint8)
    return mask, prob

def tta_predict(image_bgr, model, device, threshold=0.5):
    """
    Perform prediction with Test-Time Augmentation (TTA).

    @return: Averaged prediction mask and probability map
    """
    transforms_list = [
        lambda x: x,
        lambda x: cv2.flip(x, 1),  # Horizontal flip
        lambda x: cv2.flip(x, 0),  # Vertical flip
        lambda x: cv2.rotate(x, cv2.ROTATE_90_CLOCKWISE)
    ]
    preds = []
    probs = []
    for func in transforms_list:
        aug_image = func(image_bgr)
        mask, prob = predict_single_image(aug_image, model, device, threshold)
        # Reverse transform
        if func == transforms_list[1]:
            mask = cv2.flip(mask, 1)
            prob = cv2.flip(prob, 1)
        elif func == transforms_list[2]:
            mask = cv2.flip(mask, 0)
            prob = cv2.flip(prob, 0)
        elif func == transforms_list[3]:
            mask = cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
            prob = cv2.rotate(prob, cv2.ROTATE_90_COUNTERCLOCKWISE)
        preds.append(mask)
        probs.append(prob)
    avg_prob = np.mean(np.array(probs), axis=0)
    final_mask = (avg_prob > threshold).astype(np.uint8)
    return final_mask, avg_prob

def mask2rle(img):
    """
    Encode binary mask image into RLE (Run-Length Encoding) format.

    @param img: 2D binary mask
    @return: RLE encoded string
    """
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def main_test():
    """
    Main testing pipeline: load images, predict masks, apply DenseCRF, encode RLE, save to CSV.
    """
    root_path = "/home/cv-hacker/aaltoes-2025-computer-vision-v-1/test/"
    test_images_path = os.path.join(root_path, "images")
    submission_file = "submission.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionUNet(in_channels=3, out_channels=1)
    model.load_state_dict(torch.load("best_attention_unet.pth", map_location=device))
    model.to(device)

    submission_data = []
    image_paths = sorted(glob.glob(os.path.join(test_images_path, "*.png")))
    for img_path in image_paths:
        image_id = os.path.splitext(os.path.basename(img_path))[0]
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            print(f"Failed to read image: {img_path}")
            continue
        mask, prob = tta_predict(image_bgr, model, device, threshold=0.5)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        refined_prob = dense_crf_postprocess(image_rgb, prob)
        refined_mask = (refined_prob > 0.5).astype(np.uint8)
        encoded_pixels = mask2rle(refined_mask)
        submission_data.append({"ImageId": image_id, "EncodedPixels": encoded_pixels})

    submission_df = pd.DataFrame(submission_data, columns=["ImageId", "EncodedPixels"])
    submission_df["num"] = submission_df["ImageId"].str.extract("(\d+)", expand=False).astype(int)
    submission_df = submission_df.sort_values("num")
    submission_df.drop(columns=["num"], inplace=True)
    submission_df.to_csv(submission_file, index=False)
    print(f"Submission file saved: {submission_file}")

if __name__ == "__main__":
    main_test()
