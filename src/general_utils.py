import os
# taken from brain age data set
SCV_FILES = {'train': 'student_train_metadata.csv',
        # 'test':'student_test_metadata.csv',  images don't exist in given dataset
        'val':'student_val_metadata.csv'
        }

# this aligns with data, and assumes all data has the same internal dimensional ordering
BRAIN_PLANES = {
    'sagittal':0,
    'coronal':1,
    'axial':2
}

DL_SPLITS= ("train", "test", "val")

def prepare_environment(hpc:bool):


    if hpc:
        data_path = 'MRI_2026_datasets/brain_age'
    else:
        #set here local relative path to the data set
        data_path = '../../../../../../../mri_dataset/brain_age'

    data_path = os.path.abspath(data_path)
    print(f'Data path: {data_path}')
    os.chdir(data_path)
