import pandas as pd
from pathlib import Path
root = Path(r"C:\Users\alonz\repos\studies\mri\final_project\_report_smoke")
rows = []
for plane in ['Sagittal', 'Coronal', 'Axial']:
    for ratio in [0.2, 0.3, 0.5]:
        for i in range(3):
            sid = f'{plane[:3].lower()}_{int(ratio*100)}_{i}'
            rows.append({
                'sample_id': sid,
                'subject_id': f'subj_{i}',
                'volume_id': f'vol_{i}',
                'plane': plane,
                'slice_index': i,
                'retain_ratio': ratio,
                'mask_id': f'mask_{sid}',
                'method': 'zero_filled',
                'psnr': 20 + ratio*10 + i,
                'ssim': 0.60 + ratio*0.1 + i*0.01,
            })
            rows.append({
                'sample_id': sid,
                'subject_id': f'subj_{i}',
                'volume_id': f'vol_{i}',
                'plane': plane,
                'slice_index': i,
                'retain_ratio': ratio,
                'mask_id': f'mask_{sid}',
                'method': 'resunet',
                'psnr': 21 + ratio*10 + i,
                'ssim': 0.65 + ratio*0.1 + i*0.01,
            })
            rows.append({
                'sample_id': sid,
                'subject_id': f'subj_{i}',
                'volume_id': f'vol_{i}',
                'plane': plane,
                'slice_index': i,
                'retain_ratio': ratio,
                'mask_id': f'mask_{sid}',
                'method': 'resunet_data_consistency',
                'psnr': 22 + ratio*10 + i,
                'ssim': 0.70 + ratio*0.1 + i*0.01,
            })
pd.DataFrame(rows).to_csv(root / 'sample_metrics.csv', index=False)
print(root / 'sample_metrics.csv')
