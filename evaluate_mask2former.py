import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import evaluate
import time
from datetime import datetime
from ptflops import get_model_complexity_info
from codecarbon import EmissionsTracker

# --- Configuration ---
MODEL_ID = "facebook/mask2former-swin-small-cityscapes-semantic"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "task_1/sydneyscapes"
VAL_LIST = os.path.join(DATA_ROOT, "val.txt")
IMG_DIR = os.path.join(DATA_ROOT, "leftImg8bit", "val")
GT_DIR = os.path.join(DATA_ROOT, "gtFine", "val")
OUTPUT_DIR = "evaluation_results"

# Cityscapes label names for reporting
ID2LABEL = {
    0: "road", 1: "sidewalk", 2: "building", 3: "wall", 4: "fence", 5: "pole", 
    6: "traffic light", 7: "traffic sign", 8: "vegetation", 9: "terrain", 10: "sky", 
    11: "person", 12: "rider", 13: "car", 14: "truck", 15: "bus", 16: "train", 
    17: "motorcycle", 18: "bicycle"
}

# Cityscapes color palette for visualization
PALETTE = [
    128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
    220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
    0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32
]

def evaluate_mask2former():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. Load Model and Processor
    print(f"[*] Loading Mask2Former: {MODEL_ID}")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(MODEL_ID).to(DEVICE)
    model.eval()

    # 2. Model Parameters and Complexity
    print("[*] Calculating model complexity (GFLOPS) and parameters...")
    total_params = sum(p.numel() for p in model.parameters())
    try:
        with torch.cuda.device(DEVICE.index) if DEVICE.type == 'cuda' else torch.cpu.amp.autocast():
            macs, params = get_model_complexity_info(model, (3, 1024, 1024), as_strings=False,
                                                   print_per_layer_stat=False, verbose=False)
        gflops = macs * 2 / 1e9
    except Exception as e:
        print(f"[!] Could not calculate GFLOPS: {e}")
        gflops = 0.0

    # 3. Load Metric
    print("[*] Loading mean_iou metric...")
    metric = evaluate.load("mean_iou")

    # 4. Load Validation List
    if not os.path.exists(VAL_LIST):
        print(f"[!] Validation list not found: {VAL_LIST}")
        return

    with open(VAL_LIST, "r") as f:
        val_files = [line.strip() for line in f if line.strip()]

    print(f"[*] Starting evaluation on {len(val_files)} images on device: {DEVICE}")

    # Carbon Tracking
    print("[*] Initializing Carbon Emissions Tracker...")
    tracker = EmissionsTracker(output_dir=OUTPUT_DIR, log_level="error")
    tracker.start()

    inference_times = []
    first_image_resolution = None
    first_file_id = None
    first_pred_labels = None

    pbar = tqdm(val_files, desc="Evaluating Mask2Former", unit="image")
    for i, file_id in enumerate(pbar):
        img_path = os.path.join(IMG_DIR, f"{file_id}_leftImg8bit.png")
        gt_path = os.path.join(GT_DIR, f"{file_id}_gtFine_labelTrainIds.png")
        if not os.path.exists(img_path) or not os.path.exists(gt_path):
            continue

        image = Image.open(img_path).convert("RGB")
        target_np = np.array(Image.open(gt_path))
        
        if first_image_resolution is None:
            first_image_resolution = image.size
            first_file_id = file_id

        # Preprocess and Inference with timing
        inputs = processor(images=image, return_tensors="pt").to(DEVICE)
        
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Post-process (Mask2Former specific)
        pred_labels_tensor = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
        pred_labels = pred_labels_tensor.cpu().numpy()

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        inference_times.append(end_time - start_time)

        if i == 0:
            first_pred_labels = pred_labels

        metric.add_batch(predictions=pred_labels, references=target_np)

    # Stop Carbon Tracking
    print("\n[*] Finalizing results and carbon tracking...")
    emissions = tracker.stop()

    # 5. Final Results
    results = metric.compute(num_labels=19, ignore_index=255)
    
    if inference_times:
        avg_inference_time = sum(inference_times) / len(inference_times)
        fps = 1.0 / avg_inference_time
    else:
        avg_inference_time = 0
        fps = 0

    # Prepare Report
    report_path = os.path.join(OUTPUT_DIR, f"evaluation_report_mask2former_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(report_path, "w") as f:
        f.write("="*40 + "\n")
        f.write(f"Evaluation Results for Mask2Former ({MODEL_ID})\n")
        f.write(f"Date: {datetime.now().isoformat()}\n")
        f.write("="*40 + "\n\n")
        
        f.write(f"1. Model Info:\n")
        f.write(f"   Parameters: {total_params / 1e6:.2f} M\n")
        f.write(f"   GFLOPS (at 1024x1024): {gflops:.2f}\n\n")
        
        f.write(f"2. Performance:\n")
        if first_image_resolution:
            f.write(f"   Input Resolution: {first_image_resolution[0]}x{first_image_resolution[1]}\n")
        f.write(f"   Average Inference Time: {avg_inference_time:.4f} s\n")
        f.write(f"   FPS: {fps:.2f}\n\n")
        
        f.write(f"3. Environmental Impact:\n")
        f.write(f"   Estimated Carbon Footprint: {emissions:.6f} kg CO2\n\n")
        
        f.write(f"4. Segmentation Accuracy:\n")
        f.write(f"   Mean IoU: {results['mean_iou']:.4f}\n")
        f.write(f"   Mean Accuracy: {results['mean_accuracy']:.4f}\n")
        f.write(f"   Overall Accuracy: {results['overall_accuracy']:.4f}\n\n")
        
        f.write("Per-class IoU:\n")
        for i, iou in enumerate(results['per_category_iou']):
            label_name = ID2LABEL.get(i, f"Class {i}")
            f.write(f"  {label_name:15}: {iou:.4f}\n")

    print(f"[+] Evaluation complete. Report saved to: {report_path}")

    # 6. Save First Prediction Image
    if first_pred_labels is not None:
        safe_file_id = first_file_id.replace("/", "_").replace("\\", "_")
        mask = Image.fromarray(first_pred_labels.astype(np.uint8))
        mask.putpalette(PALETTE)
        mask_path = os.path.join(OUTPUT_DIR, f"first_prediction_mask2former_{safe_file_id}.png")
        mask.save(mask_path)
        print(f"[+] First prediction saved to: {mask_path}")

if __name__ == "__main__":
    evaluate_mask2former()
