# MMFusion AI Region Detector

**Detecting AI-Generated Image Regions via Multi-Modal Fusion Segmentation**
Final submission for the Aaltoes 2025 Computer Vision Hackathon. Achieved a **score of 0.87 Dice coefficient** on the private leaderboard.



## Competition Context

**Aaltoes 2025 Computer Vision Hackathon**
Held from **March 21 to March 23, 2025**, this competition challenges participants to develop models that can segment **AI-manipulated regions** in images. It simulates real-world problems in **media forensics, misinformation detection**, and **digital image integrity validation**.

- **Task**: Pixel-wise binary segmentation (0 = real, 1 = fake)
- **Image size**: 256 ✖️ 256 RGB
- **Training data**: 28,101 image pairs (manipulated image + binary mask)
- **Test data**: 18,735 manipulated images
- **Evaluation metric**: Dice coefficient 
- **Submission**: ust use Run-Length Encoding (RLE) format for the masks.
- **Team Work of 3 people**



## Overview

This project aims to localize AI-generated regions in images by combining multiple visual modalities and modern deep learning techniques. The final solution includes:

- **Multi-Modal Attention U-Net**: RGB + Frequency fusion
- **Test-Time Augmentation (TTA)**: Enhances robustness
- **DenseCRF Post-Processing**: Refines spatial boundaries
- **RLE Mask Encoding**: For leaderboard submission format


## Evaluation Metric: Dice Coefficient

The **Dice coefficient** (also called F1 score for segmentation) measures the overlap between the predicted mask $X$ and the ground truth mask $Y$:

$\text{Dice}(X, Y) = \frac{2 |X \cap Y|}{|X| + |Y|}$

- Score ranges from $0.00$ (no overlap) to $1.00$ (perfect overlap)


## Folder Structure

```
MMFusion AI Region Detector/
├── README.md                 Project documentation
├── train_model.py            Code used to train Model
├── main.py                   Code used to get CSV submission result
```



## Core Techniques

### 1. Multi-Modal Attention U-Net

A custom U-Net model extended with:

- **Attention Gates**: Focus on relevant regions in skip connections
- **Frequency Branch**: Extracts features from the Fourier domain (FFT magnitude)
- **Fusion Layer**: Concatenates RGB and frequency features before the bottleneck

Inspired by the [MMM 2024 paper](http://arxiv.org/abs/2312.01790): *"Exploring Multi-Modal Fusion for Image Manipulation Detection and Localization"*.

### 2. Test-Time Augmentation (TTA)

Applies 5 augmentations at inference:

- Original
- Horizontal Flip
- Vertical Flip
- 90° Rotation
- 270° Rotation

### 3. DenseCRF Refinement

Post-processes the predicted probability map to improve boundary adherence and reduce noise.

### 4. Loss Function: BCE + Dice

Ensures good balance between pixel-wise accuracy and region-level similarity.

### 5. Output Format

Submissions use Run-Length Encoding (RLE) and follow the structure:

```csv
ImageId,EncodedPixels
image_0,1 4 10 3 30 2
```



## Run the code

### Train the Model

```bash
python train_model.py
```
- Trains the `MultiModalAttentionUNet` for 50 epochs
- Saves model to `best_attention_unet.pth`

### Run Inference

```bash
python main.py
```

- Loads test images from `test/images/`
- Applies TTA and DenseCRF
- Save result as `submission.csv`



## Using a Remote GPU Server (Beginner Guide)

This project was deployed on a remote GPU server for the first time. Below are the key steps and commands used:

### 1. SSH Login to the Remote Server

```bash
ssh cv-hacker@89.169.99.20
```

If this is your first time logging in, make sure you have generated an SSH public key locally:

```bash
cat ~/.ssh/id_rsa.pub
```

Then, copy the content and add it to the server’s `~/.ssh/authorized_keys` file.

### 2. Running the Training Code in the Background

```bash
nohup python3 train_model.py > output.log 2>&1 &
```

Launching large training jobs using `nohup` can prevent interruptions from terminating the process, E.g. PC Sleep model.

To monitor the output log:

```bash
tail -f output.log
```

### 3. Monitoring GPU and System Resources

```bash
nvidia-smi      # View GPU status

top             # View real-time system resource usage
```





## Acknowledgements

- Multi-modal fusion idea inspired by MMM 2024 ([Triaridis et al.](http://arxiv.org/abs/2312.01790))
- FFT processing guided by traditional image forensics
- DenseCRF implementation adapted from Philipp Krähenbühl




