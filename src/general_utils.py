import os
# taken from brain age data set
csvs = {'train':'student_train_metadata.csv',
        'test':'student_test_metadata.csv',
        'val':'student_val_metadata.csv'
        }

# this aligns with data, and assumes all data has the same internal dimensional ordering
brain_planes = {
    'Sagittal':0,
    'Coronal':1,
    'Axial':2
}

def prepare_environment(hpc:bool):

    if hpc:
        data_path = 'MRI_2026_datasets/brain_age'
    else:
        #set local relative path to the data set
        data_path = '../../../../../../../mri_dataset/brain_age'

    data_path = os.path.abspath(data_path)
    print(f'Data path: {data_path}')
    os.chdir(data_path)