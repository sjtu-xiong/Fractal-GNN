# Fractal-GNN
Fractal-Domain Vision Graph Neural Network

File Description
(1) SPSD.py:
Fractal-domain signal processing algorithms. This file implements the SPSD2D operator used to extract local fractal spectrum histograms from image patches.
(2) model.py:
Implementation of the proposed FD-ViG (Fractal-Domain Vision Graph Neural Network), including patch embedding, fractal adapter, graph-hybrid attention encoder, and fractal diffusion propagation module.

Data Description
The model supports common remote sensing scene classification datasets such as UCMerced, RSSCN7 and SIRI-WHU.
The dataset should be organized in ImageFolder format:

dataset_root/
 ├── class_1/
 │     ├── img_1.jpg
 │     ├── img_2.jpg
 ├── class_2/
 │     ├── img_3.jpg
 │     ├── img_4.jpg


Model Usage Guide / Deployment Instructions

1. Download the following Python scripts:
- SPSD.py
- model.py

2. Install the required runtime environment:
- Python 3.9.16
- PyTorch 2.4.1
- torchvision >= 0.15
- CUDA (optional, recommended)

3. Configure model parameters in your training script:

from model import FD_ViG

model = FD_ViG(
    num_classes=21,
    img_size=256,
    patch_size=32,
    in_chans=3,
    embed_dim=192,
    num_heads=6,
    num_layers=4,
    mlp_dim=768,
    k_list=[6, 8, 8, 10],
    dropout=0.1,
    use_cls=False,
    frac_N2=9,
    frac_offset=3,
    mix_lambda_init=0.05,
    aux_init_w=0.25
)

4. Execute the training process:
Call the model forward function in your training loop:

logits = model(images)

or with auxiliary fractal supervision:

logits, aux_pred, aux_target = model(images, return_aux=True)


Operating Environment
Python 3.9.16
PyTorch 2.4.1


5. Terms of Use
This model is open-sourced under the MIT License.
You are free to use it in commercial projects.
If you use this model, we suggest (non-mandatory):
（1）Cite our paper (BibTex provided).
（2）Acknowledge the use of this model.
（3）Share your improvement experience.

Contact Us
For commercial collaboration or technical inquiries, please contact: gxiong@sjtu.edu.cn
