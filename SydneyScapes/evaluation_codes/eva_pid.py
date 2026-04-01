import os
import torch
import cv2
import numpy as np
import argparse
import time
import torch.nn.functional as F
from models.pidnet import get_pred_model
from datetime import datetime
from PIL import Image

try:
    from ptflops import get_model_complexity_info
except ImportError:
    get_model_complexity_info = None

try:
    from codecarbon import EmissionsTracker
except ImportError:
    EmissionsTracker = None

def segmentation_stats(pd, gt, num_classes):
    """混同行列の計算"""
    mask = (gt >= 0) & (gt < num_classes)
    hist = np.bincount(
        num_classes * gt[mask].astype(int) + pd[mask],
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)
    return hist

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='s', choices=['s', 'm', 'l'], help='model size')
    parser.add_argument('--weight', type=str, required=True, help='path to .pt file')
    parser.add_argument('--img_dir', type=str, required=True, help='path to leftImg8bit folder')
    parser.add_argument('--gt_dir', type=str, required=True, help='path to gtFine folder')
    parser.add_argument('--num_classes', type=int, default=19, help='number of classes')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. モデル読み込み
    model = get_pred_model(f'pidnet_{args.model}', args.num_classes)
    checkpoint = torch.load(args.weight, map_location=device)
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    
    new_state_dict = {}
    for k, v in checkpoint.items():
        name = k[6:] if k.startswith('model.') else k
        name = name[7:] if name.startswith('module.') else name
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()

    # 2. 画像とGTのペアを作成
    img_list = []
    for root, _, files in os.walk(args.img_dir):
        for f in files:
            if f.endswith('_leftImg8bit.png'):
                img_path = os.path.join(root, f)
                # 対応するGTのパスを生成 (sydneyscapes/Cityscapesの規則)
                gt_rel_path = os.path.relpath(img_path, args.img_dir)
                gt_rel_path = gt_rel_path.replace('_leftImg8bit.png', '_gtFine_labelTrainIds.png')
                gt_path = os.path.join(args.gt_dir, gt_rel_path)
                
                if os.path.exists(gt_path):
                    img_list.append((img_path, gt_path))

    if not img_list:
        print("No matching Image-GT pairs found. Check your paths.")
        return

    print(f"Total pairs found: {len(img_list)}")

    # GFLOPS and Parameters calculation
    first_img = cv2.imread(img_list[0][0])
    h_orig, w_orig, _ = first_img.shape
    
    macs_str, params_str = "N/A", "N/A"
    if get_model_complexity_info:
        try:
            with torch.cuda.device(0) if device.type == 'cuda' else torch.cpu():
                macs, params = get_model_complexity_info(model, (3, h_orig, w_orig), as_strings=True,
                                                       print_per_layer_stat=False, verbose=False)
                macs_str, params_str = macs, params
        except Exception as e:
            print(f"Error calculating complexity: {e}")
    else:
        # Fallback for parameters if ptflops is missing
        params_count = sum(p.numel() for p in model.parameters())
        params_str = f"{params_count / 1e6:.2f} M"

    # Carbon Footprint Tracker
    tracker = None
    if EmissionsTracker:
        tracker = EmissionsTracker(measure_power_secs=10, save_to_file=False)
        tracker.start()

    # 3. 推論と評価ループ
    total_hist = np.zeros((args.num_classes, args.num_classes))
    total_time = 0
    count = 0
    warmup = 10 # 最初の10枚は計測から除外

    with torch.no_grad():
        for i, (img_path, gt_path) in enumerate(img_list):
            # 画像読み込み
            img_bgr = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h, w, _ = img_rgb.shape
            
            # 前処理
            input_tensor = img_rgb.astype(np.float32) / 255.0
            input_tensor -= np.array([0.485, 0.456, 0.406])
            input_tensor /= np.array([0.229, 0.224, 0.225])
            input_tensor = input_tensor.transpose(2, 0, 1)
            input_tensor = torch.from_numpy(input_tensor).unsqueeze(0).to(device)

            # 速度計測
            torch.cuda.synchronize() if device.type == 'cuda' else None
            start_time = time.time()
            
            output = model(input_tensor)
            if isinstance(output, list):
                output = output[1]
            
            torch.cuda.synchronize() if device.type == 'cuda' else None
            elapsed = time.time() - start_time
            
            if i >= warmup:
                total_time += elapsed
                count += 1

            # 精度計測
            output = F.interpolate(output, size=(h, w), mode='bilinear', align_corners=True)
            pred_tensor = torch.argmax(output, dim=1).squeeze(0)
            pred = pred_tensor.cpu().numpy()
            
            # 最初の推論結果を保存
            if i == 0:
                palette = [
                    128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 
                    153, 153, 153, 250, 170, 30, 220, 220, 0, 107, 142, 35, 152, 251, 152, 
                    70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70, 0, 60, 100, 
                    0, 80, 100, 0, 0, 230, 119, 11, 32
                ]
                # パレットが19クラス分あるか確認 (19*3 = 57)
                while len(palette) < 256 * 3:
                    palette.append(0)
                
                res_img = Image.fromarray(pred.astype(np.uint8))
                res_img.putpalette(palette)
                res_img.save('first_prediction.png')
                print("First prediction saved as 'first_prediction.png'")

            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            total_hist += segmentation_stats(pred, gt, args.num_classes)

            if (i+1) % 50 == 0:
                print(f"Processed {i+1}/{len(img_list)} images...")

    emissions = "N/A"
    if tracker:
        emissions = tracker.stop()

    # 4. 結果表示
    output_lines = []
    output_lines.append("\n" + "="*40)
    output_lines.append(f"{'CLASS NAME':<20} | {'IoU (%)':<10}")
    output_lines.append("-" * 40)

    ID2LABEL = {
        0: "road", 1: "sidewalk", 2: "building", 3: "wall", 4: "fence", 5: "pole", 
        6: "traffic light", 7: "traffic sign", 8: "vegetation", 9: "terrain", 10: "sky", 
        11: "person", 12: "rider", 13: "car", 14: "truck", 15: "bus", 16: "train", 
        17: "motorcycle", 18: "bicycle"
    }

    # mIoU計算
    ious = np.diag(total_hist) / (total_hist.sum(axis=1) + total_hist.sum(axis=0) - np.diag(total_hist) + 1e-10)
    
    for i, iou in enumerate(ious):
        class_name = ID2LABEL.get(i, f"ID {i}")
        output_lines.append(f"{class_name:<20} | {iou*100:8.2f}%")

    miou = np.nanmean(ious) * 100
    output_lines.append("-" * 40)
    output_lines.append(f"{'OVERALL mIoU':<20} | {miou:8.2f}%")
    output_lines.append("="*40)

    # 追加メトリクス
    output_lines.append(f"\nModel: PIDNet-{args.model}")
    output_lines.append(f"Resolution: {w_orig}x{h_orig}")
    output_lines.append(f"GFLOPS (MACS): {macs_str}")
    output_lines.append(f"Parameters: {params_str}")
    
    if count > 0:
        avg_time = total_time / count
        fps = 1 / avg_time
        output_lines.append(f"Average Inference Time: {avg_time*1000:.2f} ms")
        output_lines.append(f"FPS: {fps:.2f}")
    else:
        output_lines.append("Not enough images for FPS calculation.")
    
    if emissions != "N/A":
        output_lines.append(f"Estimated Carbon Footprint: {emissions:.6f} kg CO2")
    else:
        output_lines.append("Carbon Footprint: N/A")

    # 全て出力
    final_report = "\n".join(output_lines)
    print(final_report)

    # ファイル保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"evaluation_results_{timestamp}.txt"
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(final_report)
    print(f"\nFull report saved to {report_filename}")

if __name__ == '__main__':
    main()
