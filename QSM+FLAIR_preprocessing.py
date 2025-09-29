import os
import glob
import nibabel as nib
from nipype.interfaces import fsl
from nipype.interfaces.fsl import FLIRT, Reorient2Std, FLIRT as Resample
import numpy as np
from scipy import ndimage
import multiprocessing as mp
from multiprocessing import Pool
from functools import partial
import time
from datetime import datetime

flair_path = '/NAS_248/research/DL_PRL/nnUNetv2/nnUNet_raw/Dataset008_MS_PRL_QSM_FLAIR/skull_stripped_FLAIR'
qsm_path = '/NAS_248/research/DL_PRL/nnUNetv2/nnUNet_raw/Dataset008_MS_PRL_QSM_FLAIR/QSM'
mag_path = '/NAS_248/research/DL_PRL/nnUNetv2/nnUNet_raw/Dataset008_MS_PRL_QSM_FLAIR/skull_stripped_Mag'

output_dir = '/NAS_248/research/DL_PRL/nnUNetv2/nnUNet_raw/Dataset008_MS_PRL_QSM_FLAIR/FLAIR_registered_to_QSM_250908'
matrix_dir = '/NAS_248/research/DL_PRL/nnUNetv2/nnUNet_raw/Dataset008_MS_PRL_QSM_FLAIR/registration_matrices_250908'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(matrix_dir, exist_ok=True)

flair_files = glob.glob(os.path.join(flair_path, '*.nii.gz'))
qsm_files = glob.glob(os.path.join(qsm_path, '*.nii.gz'))
mag_files = glob.glob(os.path.join(mag_path, '*.nii.gz'))

print(f"Found {len(flair_files)} FLAIR files")
print(f"Found {len(qsm_files)} QSM files")
print(f"Found {len(mag_files)} Mag files")

def extract_patient_id(filepath):
    filename = os.path.basename(filepath)
    
    if filename.startswith('MS_QSM_'):
        if '0003nii.gz' in filename:
            parts = filename.replace('MS_QSM_', '').replace('0003nii.gz', '').split('_')
            if len(parts) >= 2:
                return '_'.join(parts)
        elif filename.endswith('.nii.gz'):
            parts = filename.replace('MS_QSM_', '').replace('.nii.gz', '').split('_')
            if len(parts) >= 3:
                return '_'.join(parts[:-1])
    
    return filename.replace('.nii.gz', '').replace('nii.gz', '')

def match_files_by_patient():
    matched_files = {}
    
    for flair_file in flair_files:
        patient_id = extract_patient_id(flair_file)
        
        qsm_file = None
        mag_file = None
        
        for qsm in qsm_files:
            if extract_patient_id(qsm) == patient_id:
                qsm_file = qsm
                break
        
        for mag in mag_files:
            if extract_patient_id(mag) == patient_id:
                mag_file = mag
                break
        
        if qsm_file and mag_file:
            matched_files[patient_id] = {
                'flair': flair_file,
                'qsm': qsm_file,
                'mag': mag_file
            }
        else:
            print(f"Warning: Could not find matching files for patient {patient_id}")
    
    return matched_files

def check_image_dimensions(image_path):
    img = nib.load(image_path)
    shape = img.shape
    voxel_size = img.header.get_zooms()
    print(f"{os.path.basename(image_path)}: shape = {shape}, voxel_size = {voxel_size[:3]}")
    return shape, voxel_size

def check_image_quality(image_path):
    try:
        img = nib.load(image_path)
        data = img.get_fdata()
        
        min_val = np.min(data)
        max_val = np.max(data)
        mean_val = np.mean(data)
        std_val = np.std(data)
        
        nan_count = np.sum(np.isnan(data))
        inf_count = np.sum(np.isinf(data))
        
        print(f"Image quality check for {os.path.basename(image_path)}:")
        print(f"  Range: [{min_val:.2f}, {max_val:.2f}]")
        print(f"  Mean: {mean_val:.2f}, Std: {std_val:.2f}")
        print(f"  NaN count: {nan_count}, Inf count: {inf_count}")
        
        if nan_count > 0 or inf_count > 0:
            print(f"  WARNING: Image quality issues detected!")
            return False
        
        return True
        
    except Exception as e:
        print(f"Error checking image quality for {image_path}: {e}")
        return False

def resample_flair_to_qsm_resolution(flair_file, qsm_file, output_file):
    print(f"Resampling FLAIR to match QSM exactly...")
    
    try:
        print("Using FLIRT resampling with artifact reduction...")
        resample = FLIRT()
        resample.inputs.in_file = flair_file
        resample.inputs.reference = qsm_file
        resample.inputs.out_file = output_file
        resample.inputs.dof = 6
        resample.inputs.cost = 'mutualinfo'
        resample.inputs.interp = 'nearestneighbour'
        resample.inputs.bins = 32
        resample.inputs.no_search = True
        
        result = resample.run()
        print(f"FLIRT resampling completed: {output_file}")
        return output_file
        
    except Exception as e:
        print(f"FLIRT resampling failed: {e}")
        
        try:
            print("Using c3d reslice-identity...")
            c3d_path = "~/c3d-1.0.0-Linux-x86_64/bin/c3d"
            reslice_cmd = f"{c3d_path} -int 0 {qsm_file} {flair_file} -reslice-identity -o {output_file}"
            result = os.system(reslice_cmd)
            
            if result == 0:
                print(f"c3d resampling completed: {output_file}")
                return output_file
            else:
                print("c3d reslice command failed, trying alternative...")
                raise Exception("c3d reslice command failed")
                
        except Exception as e2:
            print(f"c3d resampling failed: {e2}")
            
            try:
                print("Using manual resampling with nibabel...")
                return manual_resample_to_target(flair_file, qsm_file, output_file)
            except Exception as e3:
                print(f"All resampling methods failed: {e3}")
                print("Returning original FLAIR file")
                return flair_file

def register_with_ants(moving_file, fixed_file, output_file, patient_id):
    try:
        print(f"ANTs registration: {os.path.basename(moving_file)} -> {os.path.basename(fixed_file)}")
        
        from nipype.interfaces.ants import Registration
        
        reg = Registration()
        reg.inputs.fixed_image = fixed_file
        reg.inputs.moving_image = moving_file
        reg.inputs.output_transform_prefix = f"{patient_id}_transform"
        reg.inputs.output_warped_image = output_file
        reg.inputs.transforms = ['Rigid', 'Affine']
        reg.inputs.transform_parameters = [(0.1,), (0.1,)]
        reg.inputs.number_of_iterations = [[1000, 500, 250, 100], [1000, 500, 250, 100]]
        reg.inputs.dimension = 3
        reg.inputs.write_composite_transform = True
        reg.inputs.collapse_output_transforms = True
        reg.inputs.initialize_transforms_per_stage = False
        reg.inputs.metric = ['MI', 'MI']
        reg.inputs.metric_weight = [1.0, 1.0]
        reg.inputs.radius_or_number_of_bins = [32, 32]
        reg.inputs.sampling_strategy = ['Regular', 'Regular']
        reg.inputs.sampling_percentage = [0.25, 0.25]
        reg.inputs.convergence_threshold = [1e-6, 1e-6]
        reg.inputs.convergence_window_size = [10, 10]
        reg.inputs.smoothing_sigmas = [[2, 1, 0], [2, 1, 0]]
        reg.inputs.sigma_units = ['vox', 'vox']
        reg.inputs.shrink_factors = [[4, 2, 1], [4, 2, 1]]
        reg.inputs.use_estimate_learning_rate_once = [True, True]
        reg.inputs.use_histogram_matching = [True, True]
        reg.inputs.winsorize_lower_quantile = 0.005
        reg.inputs.winsorize_upper_quantile = 0.995
        reg.inputs.interpolation = 'Linear'
        
        result = reg.run()
        print(f"ANTs registration completed: {output_file}")
        return output_file
        
    except Exception as e:
        print(f"ANTs registration failed: {e}")
        print("Falling back to FLIRT registration...")
        return register_with_flirt(moving_file, fixed_file, output_file, patient_id)

def apply_flair_flip(input_file, output_file):
    try:
        print(f"Applying flip to FLAIR: {os.path.basename(input_file)}")
        
        img = nib.load(input_file)
        data = img.get_fdata()
        
        print(f"Original FLAIR shape: {data.shape}")
        print(f"Original FLAIR data range: [{np.min(data):.2f}, {np.max(data):.2f}]")
        
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            print("Cleaning NaN/Inf values before flip...")
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        
        print("Applying double flip on axis=2...")
        data_flipped = np.flip(np.flip(data, axis=2), axis=2)
        
        print(f"Flipped FLAIR shape: {data_flipped.shape}")
        print(f"Flipped FLAIR data range: [{np.min(data_flipped):.2f}, {np.max(data_flipped):.2f}]")
        
        flipped_img = nib.Nifti1Image(data_flipped, img.affine, img.header.copy())
        nib.save(flipped_img, output_file)
        
        print(f"FLAIR flip completed: {output_file}")
        return output_file
        
    except Exception as e:
        print(f"Error applying flip to FLAIR {input_file}: {e}")
        import shutil
        shutil.copy2(input_file, output_file)
        return output_file

def manual_resample_to_target(source_file, target_file, output_file):
    from scipy.ndimage import zoom
    
    source_img = nib.load(source_file)
    target_img = nib.load(target_file)
    
    source_data = source_img.get_fdata()
    target_shape = target_img.shape
    target_affine = target_img.affine
    target_header = target_img.header.copy()
    
    print(f"Manual resampling: {source_data.shape} -> {target_shape}")
    
    if source_data.dtype != np.float32:
        source_data = source_data.astype(np.float32)
    
    data_min = np.min(source_data)
    data_max = np.max(source_data)
    if data_max > data_min:
        source_data = (source_data - data_min) / (data_max - data_min)
    
    scale_factors = [target_shape[i] / source_data.shape[i] for i in range(3)]
    print(f"Scale factors: {scale_factors}")
    
    resampled_data = zoom(source_data, scale_factors, order=1, mode='nearest', prefilter=False)
    
    if data_max > data_min:
        resampled_data = resampled_data * (data_max - data_min) + data_min
    
    if np.any(np.isnan(resampled_data)) or np.any(np.isinf(resampled_data)):
        print("Warning: NaN or Inf values detected, cleaning...")
        resampled_data = np.nan_to_num(resampled_data, nan=0.0, posinf=0.0, neginf=0.0)
    
    resampled_img = nib.Nifti1Image(resampled_data, target_affine, target_header)
    nib.save(resampled_img, output_file)
    
    print(f"Manual resampling completed: {resampled_data.shape}")
    return output_file

def manual_resample_to_target_smooth(source_file, target_file, output_file):
    from scipy.ndimage import zoom, gaussian_filter
    
    source_img = nib.load(source_file)
    target_img = nib.load(target_file)
    
    source_data = source_img.get_fdata()
    target_shape = target_img.shape
    target_affine = target_img.affine
    target_header = target_img.header.copy()
    
    print(f"Smooth manual resampling: {source_data.shape} -> {target_shape}")
    
    if source_data.dtype != np.float32:
        source_data = source_data.astype(np.float32)
    
    data_min = np.min(source_data)
    data_max = np.max(source_data)
    if data_max > data_min:
        source_data = (source_data - data_min) / (data_max - data_min)
    
    smoothed_data = gaussian_filter(source_data, sigma=1.0)
    
    scale_factors = [target_shape[i] / smoothed_data.shape[i] for i in range(3)]
    print(f"Scale factors: {scale_factors}")
    
    resampled_data = zoom(smoothed_data, scale_factors, order=3, mode='constant', cval=0, prefilter=True)
    
    resampled_data = gaussian_filter(resampled_data, sigma=0.5)
    
    if data_max > data_min:
        resampled_data = resampled_data * (data_max - data_min) + data_min
    
    if np.any(np.isnan(resampled_data)) or np.any(np.isinf(resampled_data)):
        print("Warning: NaN or Inf values detected, cleaning...")
        resampled_data = np.nan_to_num(resampled_data, nan=0.0, posinf=0.0, neginf=0.0)
    
    resampled_img = nib.Nifti1Image(resampled_data, target_affine, target_header)
    nib.save(resampled_img, output_file)
    
    print(f"Smooth manual resampling completed: {resampled_data.shape}")
    return output_file

def register_with_flirt(moving_file, fixed_file, output_file, patient_id):
    try:
        print(f"Starting FLIRT registration for {patient_id}")
        
        flirt = FLIRT()
        flirt.inputs.in_file = moving_file
        flirt.inputs.reference = fixed_file
        flirt.inputs.out_file = output_file
        flirt.inputs.dof = 6
        flirt.inputs.cost = 'mutualinfo'
        
        result = flirt.run()
        print(f"FLIRT registration completed for {patient_id}")
        return True
        
    except Exception as e:
        print(f"FLIRT registration failed for {patient_id}: {e}")
        return False

def process_single_patient(patient_data):
    patient_id, files = patient_data
    
    try:
        start_time = time.time()
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"\n[{timestamp}] [Process {os.getpid()}] Processing patient: {patient_id}")
        
        flair_filename = os.path.basename(files['flair'])
        final_output = os.path.join(output_dir, flair_filename)
        if os.path.exists(final_output):
            print(f"[{timestamp}] [Process {os.getpid()}] Already processed: {patient_id}")
            return f"SKIP: {patient_id}"
        
        print(f"[Process {os.getpid()}] Checking image dimensions...")
        flair_shape, flair_voxel = check_image_dimensions(files['flair'])
        qsm_shape, qsm_voxel = check_image_dimensions(files['qsm'])
        mag_shape, mag_voxel = check_image_dimensions(files['mag'])
        
        print(f"[Process {os.getpid()}] Checking image quality...")
        flair_quality = check_image_quality(files['flair'])
        qsm_quality = check_image_quality(files['qsm'])
        mag_quality = check_image_quality(files['mag'])
        
        if not (flair_quality and qsm_quality and mag_quality):
            print(f"[Process {os.getpid()}] Warning: Image quality issues detected for {patient_id}")
        
        output_file = os.path.join(output_dir, flair_filename)
        
        print(f"[Process {os.getpid()}] FLAIR size: {flair_shape}, QSM size: {qsm_shape}")
        
        temp_flair_flipped = os.path.join(matrix_dir, f'{patient_id}_flair_flipped.nii.gz')
        print(f"[Process {os.getpid()}] Step 1: Applying flip to FLAIR for {patient_id}")
        flipped_flair = apply_flair_flip(files['flair'], temp_flair_flipped)
        
        temp_reshaped_flair = os.path.join(matrix_dir, f'{patient_id}_flair_reshaped.nii.gz')
        print(f"[Process {os.getpid()}] Step 2: Reshaping FLAIR to QSM resolution for {patient_id}")
        
        reshaped_flair = resample_flair_to_qsm_resolution(flipped_flair, files['qsm'], temp_reshaped_flair)
        
        if reshaped_flair != flipped_flair:
            flair_reshaped_shape, flair_reshaped_voxel = check_image_dimensions(reshaped_flair)
            qsm_shape, qsm_voxel = check_image_dimensions(files['qsm'])
        
        print(f"[Process {os.getpid()}] Step 3: ANTs registration FLAIR to QSM for {patient_id}")
        
        try:
            register_with_ants(reshaped_flair, files['qsm'], output_file, patient_id)
            print(f"[Process {os.getpid()}] ANTs FLAIR to QSM registration completed: {output_file}")
            success = True
            
        except Exception as e:
            print(f"[Process {os.getpid()}] ANTs registration failed for {patient_id}: {e}")
            success = False
        
        if success:
            print(f"[Process {os.getpid()}] Successfully completed registration for {patient_id}: {output_file}")
            
            print(f"[Process {os.getpid()}] Checking final output quality for {patient_id}...")
            final_quality = check_image_quality(output_file)
            if not final_quality:
                print(f"[Process {os.getpid()}] WARNING: Final output has quality issues for {patient_id}")
        else:
            print(f"[Process {os.getpid()}] All registration methods failed for {patient_id}")
        
        for temp_file in [temp_flair_flipped, temp_reshaped_flair]:
            if os.path.exists(temp_file) and temp_file != files['flair'] and temp_file != files['qsm']:
                os.remove(temp_file)
        
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [Process {os.getpid()}] Completed processing for {patient_id} in {processing_time:.1f}s")
        return f"SUCCESS: {patient_id} ({processing_time:.1f}s)"
        
    except Exception as e:
        print(f"[Process {os.getpid()}] Error processing {patient_id}: {e}")
        return f"ERROR: {patient_id} - {str(e)}"

def main():
    overall_start_time = time.time()
    print("Starting FLAIR to QSM registration process with multiprocessing...")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    matched_files = match_files_by_patient()
    print(f"Matched {len(matched_files)} patients")
    
    processed_patients = []
    unprocessed_patients = []
    
    for patient_id, files in matched_files.items():
        flair_filename = os.path.basename(files['flair'])
        final_output = os.path.join(output_dir, flair_filename)
        
        if os.path.exists(final_output):
            processed_patients.append(patient_id)
        else:
            unprocessed_patients.append((patient_id, files))
    
    print(f"Already processed: {len(processed_patients)} patients")
    print(f"Need to process: {len(unprocessed_patients)} patients")
    
    if not unprocessed_patients:
        print("All patients already processed!")
        return
    
    print(f"Starting multiprocessing with 4 workers...")
    print(f"Processing {len(unprocessed_patients)} patients in parallel...")
    
    def progress_callback(result):
        print(f"Completed: {result}")
    
    with Pool(processes=4) as pool:
        async_result = pool.map_async(process_single_patient, unprocessed_patients)
        results = async_result.get()
    
    print("\n" + "="*50)
    print("PROCESSING SUMMARY")
    print("="*50)
    
    success_count = 0
    error_count = 0
    skip_count = 0
    
    for result in results:
        if result.startswith("SUCCESS"):
            success_count += 1
        elif result.startswith("ERROR"):
            error_count += 1
        elif result.startswith("SKIP"):
            skip_count += 1
    
    print(f"Successfully processed: {success_count}")
    print(f"Errors: {error_count}")
    print(f"Skipped (already processed): {skip_count}")
    print(f"Total: {len(results)}")
    
    overall_end_time = time.time()
    total_time = overall_end_time - overall_start_time
    print(f"\nTotal processing time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nRegistration process completed!")

if __name__ == "__main__":
    main()
