import glob
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage
from collections import defaultdict
import pandas as pd
import os

def analyze_lesion_sizes(mask_file, qsm_file=None, min_lesion_size=10):
    try:
        mask_img = nib.load(mask_file)
        mask_data = mask_img.get_fdata()
        
        binary_mask = (mask_data > 0).astype(np.uint8)
        labeled_mask, num_features = ndimage.label(binary_mask)
        
        lesion_sizes = []
        lesion_volumes = []
        lesion_dimensions = []
        lesion_qsm_stats = []
        
        qsm_data = None
        if qsm_file and os.path.exists(qsm_file):
            try:
                qsm_img = nib.load(qsm_file)
                qsm_data = qsm_img.get_fdata()
                print(f"QSM data loaded for {os.path.basename(mask_file)}")
            except Exception as e:
                print(f"Warning: Could not load QSM data from {qsm_file}: {e}")
                qsm_data = None
        
        if num_features > 0:
            for i in range(1, num_features + 1):
                lesion_mask = (labeled_mask == i).astype(np.uint8)
                voxel_count = np.sum(lesion_mask)
                
                if voxel_count < min_lesion_size:
                    continue
                
                voxel_volume = np.prod(mask_img.header.get_zooms())
                volume_mm3 = voxel_count * voxel_volume
                
                coords = np.where(lesion_mask > 0)
                if len(coords[0]) > 0:
                    x_range = np.max(coords[0]) - np.min(coords[0]) + 1
                    y_range = np.max(coords[1]) - np.min(coords[1]) + 1
                    z_range = np.max(coords[2]) - np.min(coords[2]) + 1
                    
                    spacing = mask_img.header.get_zooms()
                    x_size_mm = x_range * spacing[0]
                    y_size_mm = y_range * spacing[1]
                    z_size_mm = z_range * spacing[2]
                    
                    max_dimension = max(x_size_mm, y_size_mm, z_size_mm)
                    
                    lesion_sizes.append(max_dimension)
                    lesion_volumes.append(volume_mm3)
                    lesion_dimensions.append([z_size_mm, x_size_mm, y_size_mm])
                    
                    qsm_stats = None
                    if qsm_data is not None:
                        lesion_qsm_values = qsm_data[lesion_mask > 0]
                        if len(lesion_qsm_values) > 0:
                            qsm_mean = np.mean(lesion_qsm_values)
                            qsm_std = np.std(lesion_qsm_values)
                            qsm_min = np.min(lesion_qsm_values)
                            qsm_max = np.max(lesion_qsm_values)
                            qsm_median = np.median(lesion_qsm_values)
                            
                            qsm_stats = {
                                'mean': qsm_mean,
                                'std': qsm_std,
                                'min': qsm_min,
                                'max': qsm_max,
                                'median': qsm_median,
                                'mean_plus_std': qsm_mean + qsm_std,
                                'mean_minus_std': qsm_mean - qsm_std,
                                'voxel_count': len(lesion_qsm_values)
                            }
                    
                    lesion_qsm_stats.append(qsm_stats)
        
        return {
            'filename': os.path.basename(mask_file),
            'num_lesions': num_features,
            'lesion_sizes_mm': lesion_sizes,
            'lesion_volumes_mm3': lesion_volumes,
            'lesion_dimensions_mm': lesion_dimensions,
            'lesion_qsm_stats': lesion_qsm_stats,
            'total_volume_mm3': sum(lesion_volumes) if lesion_volumes else 0,
            'qsm_available': qsm_data is not None
        }
        
    except Exception as e:
        print(f"Error processing {mask_file}: {e}")
        return None

def find_qsm_file(mask_file, qsm_base_path):
    mask_basename = os.path.basename(mask_file)
    name_without_ext = mask_basename.replace('.nii.gz', '')
    qsm_filename = f"{name_without_ext}_0000.nii.gz"
    qsm_file = os.path.join(qsm_base_path, qsm_filename)
    
    return qsm_file if os.path.exists(qsm_file) else None

def analyze_all_masks(mask_paths, qsm_base_path=None, min_lesion_size=10):
    all_results = []
    
    for mask_file in mask_paths:
        print(f"Processing: {os.path.basename(mask_file)}")
        
        qsm_file = None
        if qsm_base_path:
            qsm_file = find_qsm_file(mask_file, qsm_base_path)
            if qsm_file:
                print(f"  Found QSM file: {os.path.basename(qsm_file)}")
            else:
                print(f"  QSM file not found for {os.path.basename(mask_file)}")
        
        result = analyze_lesion_sizes(mask_file, qsm_file, min_lesion_size)
        if result:
            all_results.append(result)
    
    return all_results

def calculate_statistics(all_results):
    all_sizes = []
    all_volumes = []
    all_dimensions = []
    all_qsm_stats = []
    file_stats = []
    
    for result in all_results:
        all_sizes.extend(result['lesion_sizes_mm'])
        all_volumes.extend(result['lesion_volumes_mm3'])
        all_dimensions.extend(result['lesion_dimensions_mm'])
        
        if result['qsm_available'] and result['lesion_qsm_stats']:
            for qsm_stat in result['lesion_qsm_stats']:
                if qsm_stat is not None:
                    all_qsm_stats.append(qsm_stat)
        
        file_stats.append({
            'filename': result['filename'],
            'num_lesions': result['num_lesions'],
            'total_volume_mm3': result['total_volume_mm3'],
            'avg_lesion_size_mm': np.mean(result['lesion_sizes_mm']) if result['lesion_sizes_mm'] else 0,
            'max_lesion_size_mm': np.max(result['lesion_sizes_mm']) if result['lesion_sizes_mm'] else 0,
            'min_lesion_size_mm': np.min(result['lesion_sizes_mm']) if result['lesion_sizes_mm'] else 0,
            'qsm_available': result['qsm_available']
        })
    
    if not all_sizes:
        return None, None
    
    all_dimensions = np.array(all_dimensions)
    z_sizes = all_dimensions[:, 0]
    x_sizes = all_dimensions[:, 1]
    y_sizes = all_dimensions[:, 2]
    
    stats = {
        'total_lesions': len(all_sizes),
        'size_stats': {
            'mean': np.mean(all_sizes),
            'median': np.median(all_sizes),
            'std': np.std(all_sizes),
            'min': np.min(all_sizes),
            'max': np.max(all_sizes),
            'q25': np.percentile(all_sizes, 25),
            'q75': np.percentile(all_sizes, 75)
        },
        'volume_stats': {
            'mean': np.mean(all_volumes),
            'median': np.median(all_volumes),
            'std': np.std(all_volumes),
            'min': np.min(all_volumes),
            'max': np.max(all_volumes),
            'q25': np.percentile(all_volumes, 25),
            'q75': np.percentile(all_volumes, 75)
        },
        'dimension_stats': {
            'z': {
                'mean': np.mean(z_sizes),
                'median': np.median(z_sizes),
                'std': np.std(z_sizes),
                'min': np.min(z_sizes),
                'max': np.max(z_sizes),
                'q95': np.percentile(z_sizes, 95)
            },
            'x': {
                'mean': np.mean(x_sizes),
                'median': np.median(x_sizes),
                'std': np.std(x_sizes),
                'min': np.min(x_sizes),
                'max': np.max(x_sizes),
                'q95': np.percentile(x_sizes, 95)
            },
            'y': {
                'mean': np.mean(y_sizes),
                'median': np.median(y_sizes),
                'std': np.std(y_sizes),
                'min': np.min(y_sizes),
                'max': np.max(y_sizes),
                'q95': np.percentile(y_sizes, 95)
            }
        }
    }
    
    if all_qsm_stats:
        qsm_means = [qsm['mean'] for qsm in all_qsm_stats]
        qsm_stds = [qsm['std'] for qsm in all_qsm_stats]
        qsm_mins = [qsm['min'] for qsm in all_qsm_stats]
        qsm_maxs = [qsm['max'] for qsm in all_qsm_stats]
        qsm_medians = [qsm['median'] for qsm in all_qsm_stats]
        
        stats['qsm_stats'] = {
            'mean': {
                'mean': np.mean(qsm_means),
                'std': np.std(qsm_means),
                'min': np.min(qsm_means),
                'max': np.max(qsm_means),
                'median': np.median(qsm_means)
            },
            'std': {
                'mean': np.mean(qsm_stds),
                'std': np.std(qsm_stds),
                'min': np.min(qsm_stds),
                'max': np.max(qsm_stds),
                'median': np.median(qsm_stds)
            },
            'overall': {
                'mean': np.mean(qsm_means),
                'std': np.std(qsm_means),
                'min': np.min(qsm_mins),
                'max': np.max(qsm_maxs),
                'median': np.median(qsm_medians)
            }
        }
    else:
        stats['qsm_stats'] = None
    
    return stats, file_stats

def suggest_3d_patch_size(stats):
    if not stats:
        return "No statistics data available."
    
    dim_stats = stats['dimension_stats']
    
    z_patch = int(np.ceil(dim_stats['z']['q95'] * 1.3))
    x_patch = int(np.ceil(dim_stats['x']['q95'] * 1.3))
    y_patch = int(np.ceil(dim_stats['y']['q95'] * 1.3))
    
    max_patch_size = 128
    
    z_patch = min(z_patch, max_patch_size)
    x_patch = min(x_patch, max_patch_size)
    y_patch = min(y_patch, max_patch_size)
    
    z_patch = 2 ** int(np.log2(z_patch))
    x_patch = 2 ** int(np.log2(x_patch))
    y_patch = 2 ** int(np.log2(y_patch))
    
    z_patch = max(z_patch, 32)
    x_patch = max(x_patch, 32)
    y_patch = max(y_patch, 32)
    
    total_patch_voxels = z_patch * x_patch * y_patch
    
    if total_patch_voxels <= 32**3:
        batch_size = 4
    elif total_patch_voxels <= 64**3:
        batch_size = 2
    else:
        batch_size = 1
    
    gpu_memory_gb = 8
    if gpu_memory_gb >= 16:
        batch_size = min(batch_size * 2, 8)
    elif gpu_memory_gb >= 24:
        batch_size = min(batch_size * 2, 12)
    
    suggestions = {
        'recommended_3d_patch_size': [z_patch, x_patch, y_patch],
        'patch_size_string': f"{z_patch}x{x_patch}x{y_patch}",
        'total_patch_voxels': total_patch_voxels,
        'recommended_batch_size': batch_size,
        'reasoning': {
            'z': f"Z-axis 95th percentile ({dim_stats['z']['q95']:.1f}mm) + 30% margin",
            'x': f"X-axis 95th percentile ({dim_stats['x']['q95']:.1f}mm) + 30% margin", 
            'y': f"Y-axis 95th percentile ({dim_stats['y']['q95']:.1f}mm) + 30% margin",
            'batch': f"Patch size ({total_patch_voxels} voxels) and GPU memory consideration"
        },
        'alternative_patch_sizes': [
            [64, 64, 64],
            [96, 96, 96], 
            [128, 128, 128],
            [160, 160, 160],
            [192, 192, 192]
        ],
        'alternative_batch_sizes': [1, 2, 4, 6, 8],
        'considerations': [
            f"Z-axis max lesion: {dim_stats['z']['max']:.1f}mm",
            f"X-axis max lesion: {dim_stats['x']['max']:.1f}mm", 
            f"Y-axis max lesion: {dim_stats['y']['max']:.1f}mm",
            f"GPU memory limit: {max_patch_size}",
            "Power of 2 size recommended (GPU optimization)",
            f"Expected GPU memory: {gpu_memory_gb}GB",
            "nnUNet recommendation: patch size should be 1.5-2x lesion size"
        ]
    }
    
    return suggestions

def plot_lesion_distributions(all_results, stats):
    if not all_results or not stats:
        print("No data to visualize.")
        return
    
    all_sizes = []
    all_volumes = []
    all_dimensions = []
    
    for result in all_results:
        all_sizes.extend(result['lesion_sizes_mm'])
        all_volumes.extend(result['lesion_volumes_mm3'])
        all_dimensions.extend(result['lesion_dimensions_mm'])
    
    all_dimensions = np.array(all_dimensions)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Lesion Size and Volume Analysis (3D)', fontsize=16)
    
    axes[0, 0].hist(all_sizes, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(stats['size_stats']['mean'], color='red', linestyle='--', label=f'Mean: {stats["size_stats"]["mean"]:.1f}mm')
    axes[0, 0].axvline(stats['size_stats']['median'], color='orange', linestyle='--', label=f'Median: {stats["size_stats"]["median"]:.1f}mm')
    axes[0, 0].set_xlabel('Lesion Size (mm)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Lesion Size Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].hist(all_volumes, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 1].axvline(stats['volume_stats']['mean'], color='red', linestyle='--', label=f'Mean: {stats["volume_stats"]["mean"]:.1f}mm³')
    axes[0, 1].axvline(stats['volume_stats']['median'], color='orange', linestyle='--', label=f'Median: {stats["volume_stats"]["median"]:.1f}mm³')
    axes[0, 1].set_xlabel('Lesion Volume (mm³)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Lesion Volume Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].hist(all_dimensions[:, 0], bins=30, alpha=0.7, color='coral', edgecolor='black')
    axes[0, 2].axvline(stats['dimension_stats']['z']['mean'], color='red', linestyle='--', label=f'Mean: {stats["dimension_stats"]["z"]["mean"]:.1f}mm')
    axes[0, 2].set_xlabel('Z-axis Size (mm)')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Z-axis Lesion Size Distribution')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    axes[1, 0].hist(all_dimensions[:, 1], bins=30, alpha=0.7, color='gold', edgecolor='black')
    axes[1, 0].axvline(stats['dimension_stats']['x']['mean'], color='red', linestyle='--', label=f'Mean: {stats["dimension_stats"]["x"]["mean"]:.1f}mm')
    axes[1, 0].set_xlabel('X-axis Size (mm)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('X-axis Lesion Size Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].hist(all_dimensions[:, 2], bins=30, alpha=0.7, color='plum', edgecolor='black')
    axes[1, 1].axvline(stats['dimension_stats']['y']['mean'], color='red', linestyle='--', label=f'Mean: {stats["dimension_stats"]["y"]["mean"]:.1f}mm')
    axes[1, 1].set_xlabel('Y-axis Size (mm)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Y-axis Lesion Size Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    lesion_counts = [result['num_lesions'] for result in all_results]
    axes[1, 2].hist(lesion_counts, bins=range(min(lesion_counts), max(lesion_counts) + 2), 
                     alpha=0.7, color='lightblue', edgecolor='black')
    axes[1, 2].set_xlabel('Number of Lesions per File')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].set_title('Lesion Count Distribution per File')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/NAS_248/research/DL_PRL/nnUNetv2/nnUNet_raw/Dataset006_MS_PRL_QSM/gt_lesion_analysis_3d_.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    mask_path = '/NAS_248/research/DL_PRL/nnUNetv2/nnUNet_raw/Dataset006_MS_PRL_QSM/test_gt/*.nii.gz'
    qsm_path = '/NAS_248/research/DL_PRL/nnUNetv2/nnUNet_raw/Dataset006_MS_PRL_QSM/imagesTs'
    
    mask_files = glob.glob(mask_path)
    
    if not mask_files:
        print(f"No mask files found at: {mask_path}")
        return
    
    print(f"Found {len(mask_files)} mask files to analyze.")
    print(f"QSM data path: {qsm_path}")
    
    min_lesion_size = 10
    print(f"Minimum lesion size filter: {min_lesion_size} voxels")
    print("=" * 60)
    
    all_results = analyze_all_masks(mask_files, qsm_path, min_lesion_size)
    
    if not all_results:
        print("No files available for analysis.")
        return
    
    stats, file_stats = calculate_statistics(all_results)
    
    if not stats:
        print("Cannot calculate statistics.")
        return
    
    print("\n" + "=" * 60)
    print("LESION SIZE ANALYSIS RESULTS (3D)")
    print("=" * 60)
    
    print(f"\nTotal files analyzed: {len(all_results)}")
    print(f"Total lesions found: {stats['total_lesions']}")
    
    print(f"\nLESION SIZE STATISTICS (mm):")
    print(f"  Mean: {stats['size_stats']['mean']:.2f}")
    print(f"  Median: {stats['size_stats']['median']:.2f}")
    print(f"  Std: {stats['size_stats']['std']:.2f}")
    print(f"  Min: {stats['size_stats']['min']:.2f}")
    print(f"  Max: {stats['size_stats']['max']:.2f}")
    print(f"  Q25: {stats['size_stats']['q25']:.2f}")
    print(f"  Q75: {stats['size_stats']['q75']:.2f}")
    
    print(f"\n3D DIMENSION STATISTICS (mm):")
    print(f"  Z-axis (Depth): Mean={stats['dimension_stats']['z']['mean']:.2f}, Max={stats['dimension_stats']['z']['max']:.2f}, Q95={stats['dimension_stats']['z']['q95']:.2f}")
    print(f"  X-axis (Width): Mean={stats['dimension_stats']['x']['mean']:.2f}, Max={stats['dimension_stats']['x']['max']:.2f}, Q95={stats['dimension_stats']['x']['q95']:.2f}")
    print(f"  Y-axis (Height): Mean={stats['dimension_stats']['y']['mean']:.2f}, Max={stats['dimension_stats']['y']['max']:.2f}, Q95={stats['dimension_stats']['y']['q95']:.2f}")
    
    print(f"\nLESION VOLUME STATISTICS (mm³):")
    print(f"  Mean: {stats['volume_stats']['mean']:.2f}")
    print(f"  Median: {stats['volume_stats']['median']:.2f}")
    print(f"  Std: {stats['volume_stats']['std']:.2f}")
    print(f"  Min: {stats['volume_stats']['min']:.2f}")
    print(f"  Max: {stats['volume_stats']['max']:.2f}")
    
    if stats['qsm_stats'] is not None:
        print(f"\nQSM STATISTICS (per lesion):")
        print(f"  Lesion QSM Mean:")
        print(f"    Mean: {stats['qsm_stats']['mean']['mean']:.4f}")
        print(f"    Std: {stats['qsm_stats']['mean']['std']:.4f}")
        print(f"    Min: {stats['qsm_stats']['mean']['min']:.4f}")
        print(f"    Max: {stats['qsm_stats']['mean']['max']:.4f}")
        print(f"    Median: {stats['qsm_stats']['mean']['median']:.4f}")
        
        print(f"  Lesion QSM Std:")
        print(f"    Mean: {stats['qsm_stats']['std']['mean']:.4f}")
        print(f"    Std: {stats['qsm_stats']['std']['std']:.4f}")
        print(f"    Min: {stats['qsm_stats']['std']['min']:.4f}")
        print(f"    Max: {stats['qsm_stats']['std']['max']:.4f}")
        print(f"    Median: {stats['qsm_stats']['std']['median']:.4f}")
        
        print(f"  Overall QSM (all lesion voxels):")
        print(f"    Mean: {stats['qsm_stats']['overall']['mean']:.4f}")
        print(f"    Std: {stats['qsm_stats']['overall']['std']:.4f}")
        print(f"    Min: {stats['qsm_stats']['overall']['min']:.4f}")
        print(f"    Max: {stats['qsm_stats']['overall']['max']:.4f}")
        print(f"    Median: {stats['qsm_stats']['overall']['median']:.4f}")
        
        print(f"  Lesion Mean±Std range:")
        print(f"    Mean+Std: {stats['qsm_stats']['mean']['mean'] + stats['qsm_stats']['std']['mean']:.4f}")
        print(f"    Mean-Std: {stats['qsm_stats']['mean']['mean'] - stats['qsm_stats']['std']['mean']:.4f}")
    else:
        print(f"\nQSM STATISTICS: No QSM data available")
    
    print(f"\n" + "=" * 60)
    print("3D PATCH SIZE & BATCH SIZE RECOMMENDATIONS")
    print("=" * 60)
    
    patch_suggestions = suggest_3d_patch_size(stats)
    if isinstance(patch_suggestions, dict):
        print(f"\nRecommended 3D patch size: {patch_suggestions['patch_size_string']}")
        print(f"  Z: {patch_suggestions['recommended_3d_patch_size'][0]} (Depth)")
        print(f"  X: {patch_suggestions['recommended_3d_patch_size'][1]} (Width)")
        print(f"  Y: {patch_suggestions['recommended_3d_patch_size'][2]} (Height)")
        print(f"Total patch voxels: {patch_suggestions['total_patch_voxels']:,}")
        
        print(f"\nRecommended batch size: {patch_suggestions['recommended_batch_size']}")
        
        print(f"\nReasoning:")
        print(f"  Z-axis: {patch_suggestions['reasoning']['z']}")
        print(f"  X-axis: {patch_suggestions['reasoning']['x']}")
        print(f"  Y-axis: {patch_suggestions['reasoning']['y']}")
        print(f"  Batch size: {patch_suggestions['reasoning']['batch']}")
        
        print(f"\nAlternative 3D patch sizes:")
        for alt_size in patch_suggestions['alternative_patch_sizes']:
            print(f"  {alt_size[0]}x{alt_size[1]}x{alt_size[2]}")
        
        print(f"\nAlternative batch sizes: {patch_suggestions['alternative_batch_sizes']}")
        
        print(f"\nConsiderations:")
        for consideration in patch_suggestions['considerations']:
            print(f"  - {consideration}")
    else:
        print(patch_suggestions)
    
    df = pd.DataFrame(file_stats)
    df.to_csv('/NAS_248/research/DL_PRL/nnUNetv2/nnUNet_raw/Dataset006_MS_PRL_QSM/gt_lesion_file_statistics_3d_.csv', index=False)
    print(f"\nFile statistics saved to: gt_lesion_file_statistics_3d.csv")
    
    lesion_details = []
    for result in all_results:
        for i, (size, volume, dimensions, qsm_stats) in enumerate(zip(
            result['lesion_sizes_mm'], 
            result['lesion_volumes_mm3'], 
            result['lesion_dimensions_mm'],
            result['lesion_qsm_stats']
        )):
            lesion_detail = {
                'filename': result['filename'],
                'lesion_id': i + 1,
                'size_mm': size,
                'volume_mm3': volume,
                'z_size_mm': dimensions[0],
                'x_size_mm': dimensions[1], 
                'y_size_mm': dimensions[2],
                'qsm_available': result['qsm_available']
            }
            
            if qsm_stats is not None:
                lesion_detail.update({
                    'qsm_mean': qsm_stats['mean'],
                    'qsm_std': qsm_stats['std'],
                    'qsm_min': qsm_stats['min'],
                    'qsm_max': qsm_stats['max'],
                    'qsm_median': qsm_stats['median'],
                    'qsm_mean_plus_std': qsm_stats['mean_plus_std'],
                    'qsm_mean_minus_std': qsm_stats['mean_minus_std'],
                    'qsm_voxel_count': qsm_stats['voxel_count']
                })
            else:
                lesion_detail.update({
                    'qsm_mean': None,
                    'qsm_std': None,
                    'qsm_min': None,
                    'qsm_max': None,
                    'qsm_median': None,
                    'qsm_mean_plus_std': None,
                    'qsm_mean_minus_std': None,
                    'qsm_voxel_count': None
                })
            
            lesion_details.append(lesion_detail)
    
    lesion_df = pd.DataFrame(lesion_details)
    lesion_df.to_csv('/NAS_248/research/DL_PRL/nnUNetv2/nnUNet_raw/Dataset006_MS_PRL_QSM/gt_lesion_detailed_statistics_3d_.csv', index=False)
    print(f"Detailed lesion statistics saved to: gt_lesion_detailed_statistics_3d.csv")
    
    try:
        plot_lesion_distributions(all_results, stats)
        print(f"Visualization saved to: lesion_analysis_3d.png")
    except Exception as e:
        print(f"Visualization failed: {e}")
    
    print(f"\nAnalysis completed successfully!")

if __name__ == "__main__":
    main()
