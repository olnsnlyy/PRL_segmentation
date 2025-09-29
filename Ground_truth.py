import glob
import nibabel as nib
import os
import numpy as np

def combine_segments(folder_path):
    seg1_path = os.path.join(folder_path, 'seg1.nii.gz')
    seg2_path = os.path.join(folder_path, 'seg2.nii.gz')

    if not (os.path.exists(seg1_path) and os.path.exists(seg2_path)):
        return False

    print(f"Processing folder: {folder_path}")

    try:
        seg1_img = nib.load(seg1_path)
        seg2_img = nib.load(seg2_path)

        data1 = seg1_img.get_fdata()
        data2 = seg2_img.get_fdata()

        combined_data = np.logical_or(data1, data2).astype(data1.dtype)

        folder_name = os.path.basename(os.path.normpath(folder_path))
        name_parts = folder_name.split('_')
        
        if len(name_parts) > 2:
            new_name = '_'.join(name_parts[1:-1])
            output_filename = new_name + '.nii.gz'
            output_path = os.path.join(folder_path, output_filename)

            combined_img = nib.Nifti1Image(combined_data, seg1_img.affine, seg1_img.header)
            nib.save(combined_img, output_path)
            print(f"  Saved combined mask as {output_filename}")
            return True
        else:
            print(f"  Could not generate new filename from folder: {folder_name}")
            return False

    except Exception as e:
        print(f"  Error processing files in {folder_path}: {e}")
        return False

def main():
    path = glob.glob('/NAS_248/research/DL_PRL/labels/69*/')
    
    processed_count = 0
    total_count = len(path)
    
    for folder_path in path:
        if combine_segments(folder_path):
            processed_count += 1
    
    print(f"\nProcessing completed: {processed_count}/{total_count} folders processed successfully")

if __name__ == "__main__":
    main()
