import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from django.conf import settings
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from typing import List, Dict, Tuple, Optional
from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation

def pil_to_binary_mask(pil_image: Image.Image, threshold: int = 0) -> Image.Image:
    """Convert PIL image to binary mask."""
    np_image = np.array(pil_image)
    grayscale_image = Image.fromarray(np_image).convert("L")
    binary_mask = np.array(grayscale_image) > threshold
    mask = np.zeros(binary_mask.shape, dtype=np.uint8)
    for i in range(binary_mask.shape[0]):
        for j in range(binary_mask.shape[1]):
            if binary_mask[i,j] == True:
                mask[i,j] = 1
    mask = (mask*255).astype(np.uint8)
    return Image.fromarray(mask)

class VirtualTryOn:
    def __init__(self):
        """Initialize the Virtual Try-on pipeline with all required models."""
        from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
        from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
        from src.unet_hacked_tryon import UNet2DConditionModel
        from transformers import (
            CLIPImageProcessor,
            CLIPVisionModelWithProjection,
            CLIPTextModel,
            CLIPTextModelWithProjection,
            AutoTokenizer
        )
        from diffusers import DDPMScheduler, AutoencoderKL
        from preprocess.humanparsing.run_parsing import Parsing
        from preprocess.openpose.run_openpose import OpenPose

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Base paths
        self.base_path = 'yisol/IDM-VTON'
        self.model_path = os.path.join(settings.BASE_DIR, 'virtual_tryon/IDM-VTON-hf')
        
        # Initialize models
        self.unet = UNet2DConditionModel.from_pretrained(
            self.base_path,
            subfolder="unet",
            torch_dtype=torch.float16,
        )
        self.unet.requires_grad_(False)
        
        # Initialize tokenizers
        self.tokenizer_one = AutoTokenizer.from_pretrained(
            self.base_path,
            subfolder="tokenizer",
            revision=None,
            use_fast=False,
        )
        self.tokenizer_two = AutoTokenizer.from_pretrained(
            self.base_path,
            subfolder="tokenizer_2",
            revision=None,
            use_fast=False,
        )
        
        # Initialize scheduler
        self.noise_scheduler = DDPMScheduler.from_pretrained(self.base_path, subfolder="scheduler")
        
        # Initialize encoders
        self.text_encoder_one = CLIPTextModel.from_pretrained(
            self.base_path,
            subfolder="text_encoder",
            torch_dtype=torch.float16,
        )
        self.text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
            self.base_path,
            subfolder="text_encoder_2",
            torch_dtype=torch.float16,
        )
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            self.base_path,
            subfolder="image_encoder",
            torch_dtype=torch.float16,
        )
        
        # Initialize VAE
        self.vae = AutoencoderKL.from_pretrained(
            self.base_path,
            subfolder="vae",
            torch_dtype=torch.float16,
        )
        
        # Initialize UNet Encoder
        self.unet_encoder = UNet2DConditionModel_ref.from_pretrained(
            self.base_path,
            subfolder="unet_encoder",
            torch_dtype=torch.float16,
        )
        
        # Initialize preprocessing models
        self.parsing_model = Parsing(0)
        self.openpose_model = OpenPose(0)
        
        # Set models to eval mode
        self.unet_encoder.requires_grad_(False)
        self.image_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.text_encoder_one.requires_grad_(False)
        self.text_encoder_two.requires_grad_(False)
        
        # Initialize transform
        self.tensor_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
        # Initialize pipeline
        self.pipe = TryonPipeline.from_pretrained(
            self.base_path,
            unet=self.unet,
            vae=self.vae,
            feature_extractor=CLIPImageProcessor(),
            text_encoder=self.text_encoder_one,
            text_encoder_2=self.text_encoder_two,
            tokenizer=self.tokenizer_one,
            tokenizer_2=self.tokenizer_two,
            scheduler=self.noise_scheduler,
            image_encoder=self.image_encoder,
            torch_dtype=torch.float16,
        )
        self.pipe.unet_encoder = self.unet_encoder
        
        # Move models to device
        self.move_to_device()

    def move_to_device(self):
        """Move all models to the appropriate device."""
        self.openpose_model.preprocessor.body_estimation.model.to(self.device)
        self.pipe.to(self.device)
        self.pipe.unet_encoder.to(self.device)

    def process_images(
        self,
        person_image: Image.Image,
        garment_image: Image.Image,
        garment_description: str,
        auto_mask: bool = True,
        auto_crop: bool = False,
        denoise_steps: int = 30,
        seed: int = 42
    ) -> Tuple[Image.Image, Image.Image]:
        """
        Process the try-on request with the given images and parameters.
        
        Args:
            person_image: PIL Image of the person
            garment_image: PIL Image of the garment
            garment_description: Description of the garment
            auto_mask: Whether to use auto-generated mask
            auto_crop: Whether to use auto-crop & resizing
            denoise_steps: Number of denoising steps
            seed: Random seed for generation
            
        Returns:
            Tuple of (result_image, mask_image)
        """
        # Prepare images
        garm_img = garment_image.convert("RGB").resize((768, 1024))
        human_img_orig = person_image.convert("RGB")
        
        # Handle cropping if enabled
        if auto_crop:
            width, height = human_img_orig.size
            target_width = int(min(width, height * (3 / 4)))
            target_height = int(min(height, width * (4 / 3)))
            left = (width - target_width) / 2
            top = (height - target_height) / 2
            right = (width + target_width) / 2
            bottom = (height + target_height) / 2
            cropped_img = human_img_orig.crop((left, top, right, bottom))
            crop_size = cropped_img.size
            human_img = cropped_img.resize((768, 1024))
        else:
            human_img = human_img_orig.resize((768, 1024))
        
        # Generate mask
        if auto_mask:
            keypoints = self.openpose_model(human_img.resize((384, 512)))
            model_parse, _ = self.parsing_model(human_img.resize((384, 512)))
            from utils_mask import get_mask_location
            mask, mask_gray = get_mask_location('hd', "upper_body", model_parse, keypoints)
            mask = mask.resize((768, 1024))
        else:
            # For manual mask, assuming person_image has a mask in its alpha channel
            mask = Image.new('L', (768, 1024), 0)
            # You might need to adjust this part based on how masks are provided
        
        # Process mask
        mask_gray = (1 - transforms.ToTensor()(mask)) * self.tensor_transform(human_img)
        mask_gray = to_pil_image((mask_gray + 1.0) / 2.0)
        
        # Process pose
        human_img_arg = _apply_exif_orientation(human_img.resize((384, 512)))
        human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")
        
        # Get pose image using DensePose
        import apply_net
        args = apply_net.create_argument_parser().parse_args((
            'show',
            os.path.join(self.model_path, 'configs/densepose_rcnn_R_50_FPN_s1x.yaml'),
            os.path.join(self.model_path, 'ckpt/densepose/model_final_162be9.pkl'),
            'dp_segm', '-v', '--opts', 'MODEL.DEVICE', str(self.device)
        ))
        pose_img = args.func(args, human_img_arg)
        pose_img = pose_img[:,:,::-1]
        pose_img = Image.fromarray(pose_img).resize((768, 1024))
        
        # Generate try-on image
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                # Prepare prompts
                prompt = f"model is wearing {garment_description}"
                negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
                
                # Generate embeddings
                (prompt_embeds, negative_prompt_embeds,
                 pooled_prompt_embeds, negative_pooled_prompt_embeds) = self.pipe.encode_prompt(
                    prompt,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=True,
                    negative_prompt=negative_prompt,
                )
                
                # Generate cloth embeddings
                cloth_prompt = f"a photo of {garment_description}"
                (prompt_embeds_c, _, _, _) = self.pipe.encode_prompt(
                    [cloth_prompt],
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                    negative_prompt=[negative_prompt],
                )
                
                # Prepare inputs
                pose_img = self.tensor_transform(pose_img).unsqueeze(0).to(self.device, torch.float16)
                garm_tensor = self.tensor_transform(garm_img).unsqueeze(0).to(self.device, torch.float16)
                generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None
                
                # Generate image
                images = self.pipe(
                    prompt_embeds=prompt_embeds.to(self.device, torch.float16),
                    negative_prompt_embeds=negative_prompt_embeds.to(self.device, torch.float16),
                    pooled_prompt_embeds=pooled_prompt_embeds.to(self.device, torch.float16),
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(self.device, torch.float16),
                    num_inference_steps=denoise_steps,
                    generator=generator,
                    strength=1.0,
                    pose_img=pose_img,
                    text_embeds_cloth=prompt_embeds_c.to(self.device, torch.float16),
                    cloth=garm_tensor,
                    mask_image=mask,
                    image=human_img,
                    height=1024,
                    width=768,
                    ip_adapter_image=garm_img.resize((768, 1024)),
                    guidance_scale=2.0,
                )[0]
        
        # Handle cropped image if needed
        if auto_crop:
            out_img = images[0].resize(crop_size)
            human_img_orig.paste(out_img, (int(left), int(top)))
            return human_img_orig, mask_gray
        else:
            return images[0], mask_gray

def setup_virtual_tryon() -> VirtualTryOn:
    """Initialize and return the Virtual Try-on pipeline."""
    return VirtualTryOn()