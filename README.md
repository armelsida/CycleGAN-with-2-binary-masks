# CycleGAN-with-2-binary-masks
This method is inspired from the original CycleGAN version from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix and the modified mask cycleGAN from https://github.com/minfawang/mask-cgan/blob/master/README.md.

The generators A to B and B to A were modified to take, as input, the binary mask for image
A along with image A and the binary mask for image B along with image B. The binary masks
were applied to their corresponding images using element-wise multiplication. This process
was implemented in the forward pass of the generators during training. The translated images
were then used to compute the adversarial and cycle consistency loss.
ROI_image = Real_image × Binary_mask 
where ROI_image is the output image, Real_image is the input image and Binary_mask is the
binary_mask of the input image obtained during segmentation.

![image](https://github.com/armelsida/CycleGAN-with-2-binary-masks/assets/115725362/5345302a-34a6-4eff-bf0d-53ebc073bd1b)
