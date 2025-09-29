import glob
import nibabel as nib
import numpy as np
import os
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage
from scipy.spatial.distance import directed_hausdorff
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

def load_nifti_file(file_path):
    try:
        img = nib.load(file_path)
        data = img.get_fdata()
        header = img.header
        return data, header, img.affine
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None, None

def get_voxel_spacing(affine):
    try:
        spacing = np.sqrt(np.sum(affine[:3, :3]**2, axis=0))
        return spacing
    except:
        return np.array([1.0, 1.0, 1.0])

def calculate_volume_mm3(voxel_count, voxel_spacing):
    try:
        voxel_volume = np.prod(voxel_spacing)
        return voxel_count * voxel_volume
    except:
        return voxel_count

def extract_individual_lesions(mask_data, voxel_spacing, min_lesion_size=10):
    try:
        binary_mask = (mask_data > 0).astype(np.uint8)
        labeled_mask, num_lesions = ndimage.label(binary_mask)
        
        lesions = []
        for i in range(1, num_lesions + 1):
            lesion_mask = (labeled_mask == i).astype(np.uint8)
            voxel_count = np.sum(lesion_mask)
            
            if voxel_count < min_lesion_size:
                continue
            
            try:
                centroid = ndimage.center_of_mass(lesion_mask)
                volume_mm3 = calculate_volume_mm3(voxel_count, voxel_spacing)
                
                lesion_info = {
                    'id': i,
                    'mask': lesion_mask,
                    'voxel_count': voxel_count,
                    'centroid': centroid,
                    'volume_mm3': volume_mm3
                }
                lesions.append(lesion_info)
            except Exception as e:
                print(f"  Warning: Error processing lesion {i}: {e}")
                continue
        
        return lesions, labeled_mask
        
    except Exception as e:
        print(f"Error extracting lesions: {e}")
        return [], None

def extract_lesions_by_iou_threshold(mask_data, voxel_spacing, iou_threshold=0.3, min_lesion_size=10):
    try:
        binary_mask = (mask_data > 0).astype(np.uint8)
        
        lesions = []
        visited = np.zeros_like(binary_mask, dtype=bool)
        
        window_sizes = [16, 32, 64, 128]
        
        for window_size in window_sizes:
            if window_size > min(binary_mask.shape):
                continue
                
            for z in range(0, binary_mask.shape[0] - window_size + 1, window_size // 2):
                for y in range(0, binary_mask.shape[1] - window_size + 1, window_size // 2):
                    for x in range(0, binary_mask.shape[2] - window_size + 1, window_size // 2):
                        
                        window = binary_mask[z:z+window_size, y:y+window_size, x:x+window_size]
                        
                        if np.sum(window) < min_lesion_size:
                            continue
                        
                        if np.any(visited[z:z+window_size, y:y+window_size, x:x+window_size]):
                            continue
                        
                        lesion_mask = np.zeros_like(binary_mask)
                        lesion_mask[z:z+window_size, y:y+window_size, x:x+window_size] = window
                        
                        try:
                            voxel_count = np.sum(lesion_mask)
                            if voxel_count < min_lesion_size:
                                continue
                                
                            volume_mm3 = calculate_volume_mm3(voxel_count, voxel_spacing)
                            
                            lesion_info = {
                                'voxel_count': voxel_count,
                                'volume_mm3': volume_mm3
                            }
                            lesions.append(lesion_info)
                            
                            visited[z:z+window_size, y:y+window_size, x:x+window_size] = True
                            
                        except Exception as e:
                            continue
        
        filtered_lesions = []
        for i, lesion1 in enumerate(lesions):
            is_duplicate = False
            for j, lesion2 in enumerate(filtered_lesions):
                if abs(lesion1['volume_mm3'] - lesion2['volume_mm3']) / max(lesion1['volume_mm3'], lesion2['volume_mm3']) < 0.3:
                    if lesion1['volume_mm3'] > lesion2['volume_mm3']:
                        filtered_lesions[j] = lesion1
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered_lesions.append(lesion1)
        
        return filtered_lesions
        
    except Exception as e:
        print(f"Error extracting lesions: {e}")
        return []

def calculate_3d_iou(mask1, mask2):
    try:
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        
        if union == 0:
            return 0.0
        
        return intersection / union
        
    except Exception as e:
        print(f"Error calculating 3D IoU: {e}")
        return 0.0

def calculate_3d_dice(mask1, mask2):
    try:
        intersection = np.logical_and(mask1, mask2).sum()
        union = mask1.sum() + mask2.sum()
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        return 2.0 * intersection / union
        
    except Exception as e:
        print(f"Error calculating 3D Dice: {e}")
        return 0.0

def calculate_hausdorff_distance(mask1, mask2, voxel_spacing):
    try:
        coords1 = np.where(mask1 > 0)
        coords2 = np.where(mask2 > 0)
        
        if len(coords1[0]) == 0 or len(coords2[0]) == 0:
            return np.nan
        
        points1 = np.column_stack(coords1)
        points2 = np.column_stack(coords2)
        
        points1_mm = points1 * voxel_spacing
        points2_mm = points2 * voxel_spacing
        
        hausdorff_dist = max(
            directed_hausdorff(points1_mm, points2_mm)[0],
            directed_hausdorff(points2_mm, points1_mm)[0]
        )
        
        return hausdorff_dist
        
    except Exception as e:
        print(f"Error calculating Hausdorff distance: {e}")
        return np.nan

def match_lesions(gt_lesions, pred_lesions, iou_threshold=0.3):
    try:
        matches = []
        unmatched_gt = list(range(len(gt_lesions)))
        unmatched_pred = list(range(len(pred_lesions)))
        
        iou_matrix = np.zeros((len(gt_lesions), len(pred_lesions)))
        for i, gt_lesion in enumerate(gt_lesions):
            for j, pred_lesion in enumerate(pred_lesions):
                iou = calculate_3d_iou(gt_lesion['mask'], pred_lesion['mask'])
                iou_matrix[i, j] = iou
        
        while unmatched_gt and unmatched_pred:
            max_iou = 0
            best_match = None
            
            for i in unmatched_gt:
                for j in unmatched_pred:
                    if iou_matrix[i, j] > max_iou:
                        max_iou = iou_matrix[i, j]
                        best_match = (i, j)
            
            if max_iou >= iou_threshold:
                matches.append({
                    'gt_idx': best_match[0],
                    'pred_idx': best_match[1],
                    'iou': max_iou
                })
                unmatched_gt.remove(best_match[0])
                unmatched_pred.remove(best_match[1])
            else:
                break
        
        return matches, unmatched_gt, unmatched_pred
        
    except Exception as e:
        print(f"Error matching lesions: {e}")
        return [], [], []

def calculate_lesion_metrics(gt_lesions, pred_lesions, matches, unmatched_gt, unmatched_pred, voxel_spacing):
    try:
        tp = len(matches)
        fp = len(unmatched_pred)
        fn = len(unmatched_gt)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        matched_metrics = []
        for match in matches:
            gt_lesion = gt_lesions[match['gt_idx']]
            pred_lesion = pred_lesions[match['pred_idx']]
            
            dice = calculate_3d_dice(gt_lesion['mask'], pred_lesion['mask'])
            iou = match['iou']
            hausdorff_dist = calculate_hausdorff_distance(
                gt_lesion['mask'], pred_lesion['mask'], voxel_spacing
            )
            
            volume_diff = abs(gt_lesion['volume_mm3'] - pred_lesion['volume_mm3'])
            volume_diff_percent = (volume_diff / gt_lesion['volume_mm3']) * 100 if gt_lesion['volume_mm3'] > 0 else 0
            
            matched_metrics.append({
                'gt_volume_mm3': gt_lesion['volume_mm3'],
                'pred_volume_mm3': pred_lesion['volume_mm3'],
                'volume_diff_mm3': volume_diff,
                'volume_diff_percent': volume_diff_percent,
                'dice': dice,
                'iou': iou,
                'hausdorff_distance_mm': hausdorff_dist
            })
        
        if matched_metrics:
            mean_dice = np.mean([m['dice'] for m in matched_metrics])
            mean_iou = np.mean([m['iou'] for m in matched_metrics])
            mean_hausdorff = np.mean([m['hausdorff_distance_mm'] for m in matched_metrics if not np.isnan(m['hausdorff_distance_mm'])])
            mean_volume_diff = np.mean([m['volume_diff_percent'] for m in matched_metrics])
        else:
            mean_dice = mean_iou = mean_hausdorff = mean_volume_diff = 0
        
        return {
            'detection_metrics': {
                'true_positives': tp,
                'false_positives': fp,
                'false_negatives': fn,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score
            },
            'segmentation_metrics': {
                'matched_metrics': matched_metrics,
                'mean_dice': mean_dice,
                'mean_iou': mean_iou,
                'mean_hausdorff_distance_mm': mean_hausdorff,
                'mean_volume_diff_percent': mean_volume_diff
            },
            'overall_metrics': {
                'total_gt_lesions': len(gt_lesions),
                'total_pred_lesions': len(pred_lesions),
                'matched_lesions': len(matches),
                'overall_dice': mean_dice,
                'overall_iou': mean_iou
            }
        }
        
    except Exception as e:
        print(f"Error calculating lesion metrics: {e}")
        return None

def find_matching_files(gt_dir, pred_dir):
    gt_files = glob.glob(os.path.join(gt_dir, "**/*.nii.gz"), recursive=True)
    pred_files = glob.glob(os.path.join(pred_dir, "**/*.nii.gz"), recursive=True)
    
    gt_dict = {}
    for gt_file in gt_files:
        filename = os.path.basename(gt_file)
        gt_dict[filename] = gt_file
    
    pred_dict = {}
    for pred_file in pred_files:
        filename = os.path.basename(pred_file)
        pred_dict[filename] = pred_file
    
    matched_pairs = []
    for filename in gt_dict.keys():
        if filename in pred_dict:
            matched_pairs.append({
                'filename': filename,
                'gt_path': gt_dict[filename],
                'pred_path': pred_dict[filename]
            })
    
    return matched_pairs

def process_affine_mismatch(gt_data, pred_data, gt_affine, pred_affine):
    if not np.allclose(gt_affine, pred_affine, atol=1e-6):
        print(f"  🔄 Affine mismatch detected, attempting resampling...")
        try:
            from scipy.ndimage import affine_transform
            
            source_inv = np.linalg.inv(gt_affine)
            target_inv = np.linalg.inv(pred_affine)
            transform_matrix = np.dot(target_inv, gt_affine)
            
            gt_data = affine_transform(
                gt_data,
                transform_matrix[:3, :3],
                offset=transform_matrix[:3, 3],
                output_shape=pred_data.shape,
                order=0,
                mode='constant',
                cval=0
            )
            print(f"  ✅ Resampling completed")
            return gt_data, True
        except Exception as e:
            print(f"  ❌ Resampling failed: {e}")
            return gt_data, False
    return gt_data, True

def analyze_lesion_level(matched_pairs, gt_dir, min_lesion_size=10, iou_threshold=0.3):
    lesion_results = []
    
    for pair in matched_pairs:
        filename = pair['filename']
        gt_path = pair['gt_path']
        pred_path = pair['pred_path']
        
        print(f"\nAnalyzing lesions in: {filename}")
        
        gt_data, gt_header, gt_affine = load_nifti_file(gt_path)
        pred_data, pred_header, pred_affine = load_nifti_file(pred_path)
        
        if gt_data is None or pred_data is None:
            print(f"  ⚠️  Failed to load files")
            continue
        
        gt_data, success = process_affine_mismatch(gt_data, pred_data, gt_affine, pred_affine)
        if not success:
            continue
        
        gt_spacing = get_voxel_spacing(gt_affine)
        pred_spacing = get_voxel_spacing(pred_affine)
        
        print(f"  Extracting lesions from GT...")
        gt_lesions, gt_labeled = extract_individual_lesions(gt_data, gt_spacing, min_lesion_size)
        
        print(f"  Extracting lesions from Prediction...")
        pred_lesions, pred_labeled = extract_individual_lesions(pred_data, pred_spacing, min_lesion_size)
        
        print(f"  GT lesions: {len(gt_lesions)}")
        print(f"  Pred lesions: {len(pred_lesions)}")
        
        print(f"  Matching lesions...")
        matches, unmatched_gt, unmatched_pred = match_lesions(gt_lesions, pred_lesions, iou_threshold)
        
        print(f"  Matched: {len(matches)}, Unmatched GT: {len(unmatched_gt)}, Unmatched Pred: {len(unmatched_pred)}")
        
        lesion_metrics = calculate_lesion_metrics(
            gt_lesions, pred_lesions, matches, unmatched_gt, unmatched_pred, gt_spacing
        )
        
        if lesion_metrics:
            lesion_results.append({
                'filename': filename,
                'gt_lesions': gt_lesions,
                'pred_lesions': pred_lesions,
                'matches': matches,
                'unmatched_gt': unmatched_gt,
                'unmatched_pred': unmatched_pred,
                'metrics': lesion_metrics,
                'gt_spacing': gt_spacing,
                'pred_spacing': pred_spacing
            })
        else:
            print(f"  ❌ Failed to calculate lesion metrics")
    
    return lesion_results

def analyze_patient_level(matched_pairs, gt_dir, min_lesion_size=10, iou_threshold=0.3):
    patient_results = []
    
    for pair in matched_pairs:
        filename = pair['filename']
        gt_path = pair['gt_path']
        pred_path = pair['pred_path']
        
        print(f"\nAnalyzing patient: {filename}")
        
        gt_data, gt_header, gt_affine = load_nifti_file(gt_path)
        pred_data, pred_header, pred_affine = load_nifti_file(pred_path)
        
        if gt_data is None or pred_data is None:
            print(f"  ⚠️  Failed to load files")
            continue
        
        gt_data, success = process_affine_mismatch(gt_data, pred_data, gt_affine, pred_affine)
        if not success:
            continue
        
        gt_spacing = get_voxel_spacing(gt_affine)
        pred_spacing = get_voxel_spacing(pred_affine)
        
        print(f"  Extracting lesions from GT...")
        gt_lesions = extract_lesions_by_iou_threshold(gt_data, gt_spacing, iou_threshold, min_lesion_size)
        
        print(f"  Extracting lesions from Prediction...")
        pred_lesions = extract_lesions_by_iou_threshold(pred_data, pred_spacing, iou_threshold, min_lesion_size)
        
        print(f"  GT lesions: {len(gt_lesions)}")
        print(f"  Pred lesions: {len(pred_lesions)}")
        
        gt_has_lesion = len(gt_lesions) > 0
        pred_has_lesion = len(pred_lesions) > 0
        
        gt_lesion_count = len(gt_lesions)
        pred_lesion_count = len(pred_lesions)
        
        matched_lesions = 0
        for gt_lesion in gt_lesions:
            gt_volume = gt_lesion['volume_mm3']
            min_volume_diff = float('inf')
            for pred_lesion in pred_lesions:
                pred_volume = pred_lesion['volume_mm3']
                volume_diff = abs(gt_volume - pred_volume)
                if volume_diff < min_volume_diff and volume_diff < gt_volume * 0.5:
                    min_volume_diff = volume_diff
            
            if min_volume_diff != float('inf'):
                matched_lesions += 1
        
        lesion_count_diff = pred_lesion_count - gt_lesion_count
        lesion_count_ratio = pred_lesion_count / gt_lesion_count if gt_lesion_count > 0 else np.nan
        matched_lesion_ratio = matched_lesions / gt_lesion_count if gt_lesion_count > 0 else np.nan
        
        gt_total_volume = sum(lesion['volume_mm3'] for lesion in gt_lesions)
        pred_total_volume = sum(lesion['volume_mm3'] for lesion in pred_lesions)
        volume_diff = pred_total_volume - gt_total_volume
        volume_ratio = pred_total_volume / gt_total_volume if gt_total_volume > 0 else np.nan
        
        patient_result = {
            'filename': filename,
            'gt_has_lesion': gt_has_lesion,
            'pred_has_lesion': pred_has_lesion,
            'gt_lesion_count': gt_lesion_count,
            'pred_lesion_count': pred_lesion_count,
            'matched_lesion_count': matched_lesions,
            'lesion_count_diff': lesion_count_diff,
            'lesion_count_ratio': lesion_count_ratio,
            'matched_lesion_ratio': matched_lesion_ratio,
            'gt_total_volume': gt_total_volume,
            'pred_total_volume': pred_total_volume,
            'volume_diff': volume_diff,
            'volume_ratio': volume_ratio
        }
        
        patient_results.append(patient_result)
    
    return patient_results

def calculate_binary_metrics(patient_results):
    tp = sum(1 for p in patient_results if p['gt_has_lesion'] and p['pred_has_lesion'])
    tn = sum(1 for p in patient_results if not p['gt_has_lesion'] and not p['pred_has_lesion'])
    fp = sum(1 for p in patient_results if not p['gt_has_lesion'] and p['pred_has_lesion'])
    fn = sum(1 for p in patient_results if p['gt_has_lesion'] and not p['pred_has_lesion'])
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    return {
        'confusion_matrix': {
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
        },
        'metrics': {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'ppv': ppv,
            'npv': npv,
            'accuracy': accuracy
        }
    }

def calculate_correlation_metrics(patient_results):
    gt_counts = [p['gt_lesion_count'] for p in patient_results]
    matched_counts = [p['matched_lesion_count'] for p in patient_results]
    
    count_pearson_r, count_pearson_p = pearsonr(gt_counts, matched_counts) if len(gt_counts) > 1 else (np.nan, np.nan)
    count_spearman_r, count_spearman_p = spearmanr(gt_counts, matched_counts) if len(gt_counts) > 1 else (np.nan, np.nan)
    
    gt_volumes = [p['gt_total_volume'] for p in patient_results]
    pred_volumes = [p['pred_total_volume'] for p in patient_results]
    
    volume_pearson_r, volume_pearson_p = pearsonr(gt_volumes, pred_volumes) if len(gt_volumes) > 1 else (np.nan, np.nan)
    volume_spearman_r, volume_spearman_p = spearmanr(gt_volumes, pred_volumes) if len(gt_volumes) > 1 else (np.nan, np.nan)
    
    return {
        'matched_lesion_count_correlation': {
            'pearson_r': count_pearson_r,
            'pearson_p': count_pearson_p,
            'spearman_r': count_spearman_r,
            'spearman_p': count_spearman_p
        },
        'volume_correlation': {
            'pearson_r': volume_pearson_r,
            'pearson_p': volume_pearson_p,
            'spearman_r': volume_spearman_r,
            'spearman_p': volume_spearman_p
        }
    }

def save_lesion_results(lesion_results, output_dir):
    if not lesion_results:
        print("No lesion results to save.")
        return
    
    detection_summary = []
    for result in lesion_results:
        filename = result['filename']
        metrics = result['metrics']
        
        detection_summary.append({
            'filename': filename,
            'true_positives': metrics['detection_metrics']['true_positives'],
            'false_positives': metrics['detection_metrics']['false_positives'],
            'false_negatives': metrics['detection_metrics']['false_negatives'],
            'precision': metrics['detection_metrics']['precision'],
            'recall': metrics['detection_metrics']['recall'],
            'f1_score': metrics['detection_metrics']['f1_score']
        })
    
    df_detection = pd.DataFrame(detection_summary)
    detection_csv_path = os.path.join(output_dir, 'detection_performance.csv')
    df_detection.to_csv(detection_csv_path, index=False)
    print(f"Detection performance saved to: {detection_csv_path}")
    
    segmentation_metrics = []
    for result in lesion_results:
        filename = result['filename']
        for metric in result['metrics']['segmentation_metrics']['matched_metrics']:
            metric['filename'] = filename
            segmentation_metrics.append(metric)
    
    if segmentation_metrics:
        df_segmentation = pd.DataFrame(segmentation_metrics)
        segmentation_csv_path = os.path.join(output_dir, 'segmentation_performance.csv')
        df_segmentation.to_csv(segmentation_csv_path, index=False)
        print(f"Segmentation performance saved to: {segmentation_csv_path}")
    
    overall_summary = []
    for result in lesion_results:
        filename = result['filename']
        metrics = result['metrics']
        
        overall_summary.append({
            'filename': filename,
            'total_gt_lesions': metrics['overall_metrics']['total_gt_lesions'],
            'total_pred_lesions': metrics['overall_metrics']['total_pred_lesions'],
            'matched_lesions': metrics['overall_metrics']['matched_lesions'],
            'overall_dice': metrics['overall_metrics']['overall_dice'],
            'overall_iou': metrics['overall_metrics']['overall_iou'],
            'mean_hausdorff_distance_mm': metrics['segmentation_metrics']['mean_hausdorff_distance_mm'],
            'mean_volume_diff_percent': metrics['segmentation_metrics']['mean_volume_diff_percent']
        })
    
    df_overall = pd.DataFrame(overall_summary)
    overall_csv_path = os.path.join(output_dir, 'overall_performance.csv')
    df_overall.to_csv(overall_csv_path, index=False)
    print(f"Overall performance saved to: {overall_csv_path}")
    
    return df_detection, df_segmentation, df_overall

def save_patient_results(patient_results, binary_metrics, correlation_metrics, output_dir):
    if not patient_results:
        print("No patient results to save.")
        return
    
    df_patient = pd.DataFrame(patient_results)
    patient_csv_path = os.path.join(output_dir, 'patient_level_results.csv')
    df_patient.to_csv(patient_csv_path, index=False)
    print(f"Patient-level results saved to: {patient_csv_path}")
    
    binary_summary = {
        'metric': ['Sensitivity', 'Specificity', 'PPV', 'NPV', 'Accuracy'],
        'value': [
            binary_metrics['metrics']['sensitivity'],
            binary_metrics['metrics']['specificity'],
            binary_metrics['metrics']['ppv'],
            binary_metrics['metrics']['npv'],
            binary_metrics['metrics']['accuracy']
        ]
    }
    df_binary = pd.DataFrame(binary_summary)
    binary_csv_path = os.path.join(output_dir, 'binary_detection_summary.csv')
    df_binary.to_csv(binary_csv_path, index=False)
    print(f"Binary detection summary saved to: {binary_csv_path}")
    
    correlation_summary = {
        'metric': [
            'Matched Lesion Count Pearson r', 'Matched Lesion Count Pearson p',
            'Matched Lesion Count Spearman r', 'Matched Lesion Count Spearman p',
            'Volume Pearson r', 'Volume Pearson p',
            'Volume Spearman r', 'Volume Spearman p'
        ],
        'value': [
            correlation_metrics['matched_lesion_count_correlation']['pearson_r'],
            correlation_metrics['matched_lesion_count_correlation']['pearson_p'],
            correlation_metrics['matched_lesion_count_correlation']['spearman_r'],
            correlation_metrics['matched_lesion_count_correlation']['spearman_p'],
            correlation_metrics['volume_correlation']['pearson_r'],
            correlation_metrics['volume_correlation']['pearson_p'],
            correlation_metrics['volume_correlation']['spearman_r'],
            correlation_metrics['volume_correlation']['spearman_p']
        ]
    }
    df_correlation = pd.DataFrame(correlation_summary)
    correlation_csv_path = os.path.join(output_dir, 'correlation_summary.csv')
    df_correlation.to_csv(correlation_csv_path, index=False)
    print(f"Correlation summary saved to: {correlation_csv_path}")
    
    return df_patient, df_binary, df_correlation

def plot_lesion_analysis(lesion_results, output_dir):
    try:
        if not lesion_results:
            print("No data to visualize.")
            return
        
        all_precisions = []
        all_recalls = []
        all_f1_scores = []
        all_dice_scores = []
        all_iou_scores = []
        all_hausdorff_dists = []
        all_volume_diffs = []
        
        for result in lesion_results:
            metrics = result['metrics']
            all_precisions.append(metrics['detection_metrics']['precision'])
            all_recalls.append(metrics['detection_metrics']['recall'])
            all_f1_scores.append(metrics['detection_metrics']['f1_score'])
            
            if metrics['segmentation_metrics']['matched_metrics']:
                all_dice_scores.extend([m['dice'] for m in metrics['segmentation_metrics']['matched_metrics']])
                all_iou_scores.extend([m['iou'] for m in metrics['segmentation_metrics']['matched_metrics']])
                all_hausdorff_dists.extend([m['hausdorff_distance_mm'] for m in metrics['segmentation_metrics']['matched_metrics'] if not np.isnan(m['hausdorff_distance_mm'])])
                all_volume_diffs.extend([m['volume_diff_percent'] for m in metrics['segmentation_metrics']['matched_metrics']])
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Lesion-Level Analysis Results', fontsize=16)
        
        axes[0, 0].scatter(all_precisions, all_recalls, alpha=0.7, color='blue')
        axes[0, 0].set_xlabel('Precision')
        axes[0, 0].set_ylabel('Recall')
        axes[0, 0].set_title('Detection Performance')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].hist(all_f1_scores, bins=20, alpha=0.7, color='green', edgecolor='black')
        axes[0, 1].axvline(np.mean(all_f1_scores), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(all_f1_scores):.3f}')
        axes[0, 1].set_xlabel('F1 Score')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('F1 Score Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        if all_dice_scores:
            axes[0, 2].hist(all_dice_scores, bins=20, alpha=0.7, color='orange', edgecolor='black')
            axes[0, 2].axvline(np.mean(all_dice_scores), color='red', linestyle='--', 
                               label=f'Mean: {np.mean(all_dice_scores):.3f}')
            axes[0, 2].set_xlabel('Dice Score')
            axes[0, 2].set_ylabel('Frequency')
            axes[0, 2].set_title('Dice Score Distribution')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        if all_iou_scores:
            axes[1, 0].hist(all_iou_scores, bins=20, alpha=0.7, color='purple', edgecolor='black')
            axes[1, 0].axvline(np.mean(all_iou_scores), color='red', linestyle='--', 
                               label=f'Mean: {np.mean(all_iou_scores):.3f}')
            axes[1, 0].set_xlabel('IoU Score')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('IoU Score Distribution')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        if all_hausdorff_dists:
            axes[1, 1].hist(all_hausdorff_dists, bins=20, alpha=0.7, color='brown', edgecolor='black')
            axes[1, 1].axvline(np.mean(all_hausdorff_dists), color='red', linestyle='--', 
                               label=f'Mean: {np.mean(all_hausdorff_dists):.2f}mm')
            axes[1, 1].set_xlabel('Hausdorff Distance (mm)')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Hausdorff Distance Distribution')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        if all_volume_diffs:
            axes[1, 2].hist(all_volume_diffs, bins=20, alpha=0.7, color='pink', edgecolor='black')
            axes[1, 2].axvline(np.mean(all_volume_diffs), color='red', linestyle='--', 
                               label=f'Mean: {np.mean(all_volume_diffs):.1f}%')
            axes[1, 2].set_xlabel('Volume Difference (%)')
            axes[1, 2].set_ylabel('Frequency')
            axes[1, 2].set_title('Volume Difference Distribution')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'lesion_level_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Visualization saved to: {plot_path}")
        
    except Exception as e:
        print(f"Visualization failed: {e}")

def plot_patient_analysis(patient_results, binary_metrics, correlation_metrics, output_dir):
    try:
        if not patient_results:
            print("No data to visualize.")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Patient-Level Analysis Results', fontsize=16)
        
        metrics = binary_metrics['metrics']
        metric_names = ['Sensitivity', 'Specificity', 'PPV', 'NPV', 'Accuracy']
        metric_values = [metrics['sensitivity'], metrics['specificity'], metrics['ppv'], metrics['npv'], metrics['accuracy']]
        
        bars = axes[0, 0].bar(metric_names, metric_values, color=['blue', 'green', 'orange', 'red', 'purple'], alpha=0.7)
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Binary Detection Performance')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].grid(True, alpha=0.3)
        
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        gt_counts = [p['gt_lesion_count'] for p in patient_results]
        matched_counts = [p['matched_lesion_count'] for p in patient_results]
        
        axes[0, 1].scatter(gt_counts, matched_counts, alpha=0.7, color='blue')
        axes[0, 1].plot([0, max(max(gt_counts), max(matched_counts))], [0, max(max(gt_counts), max(matched_counts))], 'r--', alpha=0.7)
        axes[0, 1].set_xlabel('Ground Truth Lesion Count')
        axes[0, 1].set_ylabel('Matched Lesion Count')
        axes[0, 1].set_title(f'Matched Lesion Count Correlation\nPearson r: {correlation_metrics["matched_lesion_count_correlation"]["pearson_r"]:.3f}')
        axes[0, 1].grid(True, alpha=0.3)
        
        gt_volumes = [p['gt_total_volume'] for p in patient_results]
        pred_volumes = [p['pred_total_volume'] for p in patient_results]
        
        axes[0, 2].scatter(gt_volumes, pred_volumes, alpha=0.7, color='green')
        axes[0, 2].plot([0, max(max(gt_volumes), max(pred_volumes))], [0, max(max(gt_volumes), max(pred_volumes))], 'r--', alpha=0.7)
        axes[0, 2].set_xlabel('Ground Truth Total Volume (mm³)')
        axes[0, 2].set_ylabel('Predicted Total Volume (mm³)')
        axes[0, 2].set_title(f'Total Volume Correlation\nPearson r: {correlation_metrics["volume_correlation"]["pearson_r"]:.3f}')
        axes[0, 2].grid(True, alpha=0.3)
        
        count_diffs = [p['lesion_count_diff'] for p in patient_results]
        axes[1, 0].hist(count_diffs, bins=20, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 0].axvline(np.mean(count_diffs), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(count_diffs):.2f}')
        axes[1, 0].set_xlabel('Lesion Count Difference (Pred - GT)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Lesion Count Difference Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        volume_diffs = [p['volume_diff'] for p in patient_results]
        axes[1, 1].hist(volume_diffs, bins=20, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 1].axvline(np.mean(volume_diffs), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(volume_diffs):.1f}mm³')
        axes[1, 1].set_xlabel('Volume Difference (Pred - GT, mm³)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Volume Difference Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        cm = binary_metrics['confusion_matrix']
        cm_matrix = np.array([[cm['tn'], cm['fp']], [cm['fn'], cm['tp']]])
        
        im = axes[1, 2].imshow(cm_matrix, cmap='Blues', alpha=0.8)
        axes[1, 2].set_xticks([0, 1])
        axes[1, 2].set_yticks([0, 1])
        axes[1, 2].set_xticklabels(['No Lesion', 'Has Lesion'])
        axes[1, 2].set_yticklabels(['No Lesion', 'Has Lesion'])
        axes[1, 2].set_title('Confusion Matrix')
        
        for i in range(2):
            for j in range(2):
                axes[1, 2].text(j, i, str(cm_matrix[i, j]), ha='center', va='center', fontsize=14, color='black')
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'patient_level_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Visualization saved to: {plot_path}")
        
    except Exception as e:
        print(f"Visualization failed: {e}")

def main():
    gt_dir = "/NAS_248/research/DL_PRL/nnUNetv2/nnUNet_raw/Dataset007_MS_PRL_QSM_Mag_Pha/test_gt"
    pred_dir = "/NAS_248/research/DL_PRL/nnUNetv2/nnUNet_raw/Dataset007_MS_PRL_QSM_Mag_Pha/outputsTs"
    output_dir = "/NAS_248/research/DL_PRL/nnUNetv2/20250903/D007_evaluation_results"
    
    min_lesion_size = 10
    iou_threshold = 0.3
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Ground Truth Directory: {gt_dir}")
    print(f"Prediction Directory: {pred_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Minimum Lesion Size: {min_lesion_size} voxels")
    print(f"IoU Threshold: {iou_threshold}")
    print("=" * 60)
    
    if not os.path.exists(gt_dir):
        print(f"Error: Ground truth directory {gt_dir} does not exist!")
        return
    
    if not os.path.exists(pred_dir):
        print(f"Error: Prediction directory {pred_dir} does not exist!")
        return
    
    print("Finding matching files...")
    matched_pairs = find_matching_files(gt_dir, pred_dir)
    
    if not matched_pairs:
        print("No matching files found!")
        return
    
    print(f"Found {len(matched_pairs)} matching file pairs")
    
    print("\nStarting lesion-level analysis...")
    lesion_results = analyze_lesion_level(matched_pairs, gt_dir, min_lesion_size, iou_threshold)
    
    print("\nStarting patient-level analysis...")
    patient_results = analyze_patient_level(matched_pairs, gt_dir, min_lesion_size, iou_threshold)
    
    if not lesion_results and not patient_results:
        print("No results generated.")
        return
    
    print("\nSaving lesion-level results...")
    if lesion_results:
        save_lesion_results(lesion_results, output_dir)
        
        print("\nGenerating lesion-level visualizations...")
        plot_lesion_analysis(lesion_results, output_dir)
    
    print("\nCalculating patient-level metrics...")
    if patient_results:
        binary_metrics = calculate_binary_metrics(patient_results)
        correlation_metrics = calculate_correlation_metrics(patient_results)
        
        print("\nSaving patient-level results...")
        save_patient_results(patient_results, binary_metrics, correlation_metrics, output_dir)
        
        print("\nGenerating patient-level visualizations...")
        plot_patient_analysis(patient_results, binary_metrics, correlation_metrics, output_dir)
    
    print(f"\n{'='*60}")
    print("MODEL EVALUATION COMPLETED!")
    print(f"{'='*60}")
    print(f"Total files analyzed: {len(matched_pairs)}")
    
    if lesion_results:
        mean_precision = np.mean([r['metrics']['detection_metrics']['precision'] for r in lesion_results])
        mean_recall = np.mean([r['metrics']['detection_metrics']['recall'] for r in lesion_results])
        mean_f1 = np.mean([r['metrics']['detection_metrics']['f1_score'] for r in lesion_results])
        
        print(f"\nLesion-Level Detection Performance (Average):")
        print(f"  Precision: {mean_precision:.3f}")
        print(f"  Recall: {mean_recall:.3f}")
        print(f"  F1-Score: {mean_f1:.3f}")
    
    if patient_results:
        print(f"\nPatient-Level Detection Performance:")
        print(f"  Sensitivity: {binary_metrics['metrics']['sensitivity']:.3f}")
        print(f"  Specificity: {binary_metrics['metrics']['specificity']:.3f}")
        print(f"  PPV: {binary_metrics['metrics']['ppv']:.3f}")
        print(f"  NPV: {binary_metrics['metrics']['npv']:.3f}")
        print(f"  Accuracy: {binary_metrics['metrics']['accuracy']:.3f}")
    
    print(f"\nResults saved in '{output_dir}' directory.")

if __name__ == "__main__":
    main()
