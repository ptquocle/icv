import os
import gc
import pandas as pd
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from tqdm import tqdm

import config
from model import DenseNet14
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

def get_gradcam_image(model, img_tensor, original_rgb, target_class_idx):
    """Generates a Grad-CAM overlay for a specific predicted class."""
    target_layers = [model.densenet.features[-1]]
    
    cam = GradCAM(model=model, target_layers=target_layers)
    
    targets = [ClassifierOutputTarget(target_class_idx)]
    grayscale_cam = cam(input_tensor=img_tensor, targets=targets)[0, :]
    
    visualization = show_cam_on_image(original_rgb, grayscale_cam, use_rgb=True)
    return Image.fromarray(visualization)

def run_batch_generation(csv_path, 
                         model_checkpoint=os.path.join(BASE_DIR, "checkpoints", "best_model.pth"),
                         output_csv=os.path.join(BASE_DIR, "results", "dataset_explanation")):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    combined_images_dir = os.path.join(BASE_DIR, "results", "combined_images")
    os.makedirs(combined_images_dir, exist_ok=True)
    
    df = pd.read_csv(csv_path)

    # PASS 1: CLASSIFIER & GRAD-CAM GENERATION
    classifier = DenseNet14().to(device)
    classifier.load_state_dict(torch.load(model_checkpoint, map_location=device))
    classifier.eval()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    batch_metadata = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing Grad-CAMs"):
        try:
            sub_id = str(row['subject_id']).strip()
            subject_id = sub_id if sub_id.startswith('p') else f"p{sub_id}"
            
            st_id = str(row['study_id']).strip()
            study_id = st_id if st_id.startswith('s') else f"s{st_id}"
            
            dicom_id = str(row['dicom_id']).strip().replace(".jpg", "")
            
            partition = subject_id[:3]
            
            image_path = os.path.join(
                BASE_DIR,
                "data",
                "files",
                partition, 
                subject_id, 
                study_id, 
                f"{dicom_id}.jpg"
            )

            if not os.path.exists(image_path):
                continue

            raw_img = Image.open(image_path).convert('RGB')
            raw_img = raw_img.resize((config.IMAGE_SIZE, config.IMAGE_SIZE))
            img_float_np = np.float32(raw_img) / 255.0
            img_tensor = preprocess(raw_img).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = torch.sigmoid(classifier(img_tensor))
            
            confidence, predicted_idx = torch.max(outputs, 1)
            predicted_class = config.CLASSES[predicted_idx.item()]

            heatmap_img = get_gradcam_image(classifier, img_tensor, img_float_np, predicted_idx.item())

            # Compile side-by-side prompt setup (Original | Heatmap)
            combined_img = Image.new('RGB', (raw_img.width * 2, raw_img.height))
            combined_img.paste(raw_img, (0, 0))
            combined_img.paste(heatmap_img, (raw_img.width, 0))

            temp_img_path = os.path.join(combined_images_dir, f"temp_{idx}.png")
            combined_img.save(temp_img_path)

            batch_metadata.append({
                'index': idx,
                'dicom_id': dicom_id,
                'study_id': study_id,
                'subject_id': subject_id,
                'temp_image_path': temp_img_path,
                'predicted_class': predicted_class,
                'confidence': confidence.item()
            })
            
        except Exception as e:
            print(f"Skipping row index {idx} due to calculation error: {e}")

    # Dump Pass 1 from memory blocks to free up GPU allocations
    del classifier
    if 'img_tensor' in locals(): del img_tensor
    if 'outputs' in locals(): del outputs
    gc.collect()
    torch.cuda.empty_cache()

    if not batch_metadata:
        return

    # PASS 2: VLM TEXT EXPLANATION GENERATION
    vlm_model_id = "Qwen/Qwen2-VL-7B-Instruct"
    
    processor = AutoProcessor.from_pretrained(vlm_model_id)
    vlm = Qwen2VLForConditionalGeneration.from_pretrained(
        vlm_model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    final_results = []

    for meta in tqdm(batch_metadata, desc="Generating Explanations"):
        try:
            combined_img = Image.open(meta['temp_image_path']).convert('RGB')
            
            prompt_instruction = (
                f"The image on the left is a chest X-ray. The image on the right shows a Grad-CAM heatmap "
                f"generated by an AI that diagnosed '{meta['predicted_class']}'. Act as an expert radiologist. "
                f"Based only on the anatomical structures highlighted in red/yellow in the heatmap on the right, "
                f"write a 2-sentence meaningful clinical explanation of the visual evidence for this diagnosis, using relevant clinical terms. Do not make unnecessary repetition."
            )

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": combined_img},
                        {"type": "text", "text": prompt_instruction}
                    ]
                }
            ]

            prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[prompt_text], images=[combined_img], padding=True, return_tensors="pt").to(device)

            output = vlm.generate(**inputs, max_new_tokens=100, temperature=0.2)
            
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output)]
            explanation = processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0].strip()

            meta['explanation'] = explanation
            final_results.append(meta)

            saved_df = pd.DataFrame(final_results)
            saved_df.to_csv(output_csv, index=False)

            os.remove(meta['temp_image_path'])

        except Exception as e:
            print(f"Error executing text processing generation at index {meta['index']}: {e}")

    try:
        os.rmdir("../results/combined_images")
    except Exception:
        pass


if __name__ == "__main__":
    target_split_csv = os.path.join(BASE_DIR, "data", "mimic_train.csv")
    train_output_csv = os.path.join(BASE_DIR, "results", "train_explanations.csv")
    run_batch_generation(target_split_csv)
