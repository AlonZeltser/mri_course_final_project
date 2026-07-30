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

