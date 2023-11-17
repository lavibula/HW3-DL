# HW3-DL

## Instructions
```python
import requests
import os

# Replace YOUR_DRIVE_URL with the direct link to the Google Drive file
drive_url = 'YOUR_DRIVE_URL'
''''
https://drive.google.com/file/d/1auw05aPCy8i0vnqi16gXfkQxtpGDVQZh/view?usp=sharing

or

https://drive.google.com/u/0/uc?id=1auw05aPCy8i0vnqi16gXfkQxtpGDVQZh&export=download
'''
# Directory where the downloaded file will be saved
save_dir = '/kaggle/working/'

# Send a GET request to the drive_url
response = requests.get(drive_url)

# Write the content of the response to a file in the save_dir
with open(os.path.join(save_dir, 'model.pth'), 'wb') as f:
    f.write(response.content)
```
```python
!git clone https://github.com/lavibula/HW3-DL.git # clone my git repo
```
```python
!mkdir predicted_mask # make dir for mask prediction
```
```python
!python /kaggle/working/BKAI_Polyp/infer.py --checkpoint '/kaggle/working/unet_model.pth' --test_dir '/kaggle/input/bkai-igh-neopolyp/test/test' --mask_dir '/kaggle/working/prediction'

# parse args checkpoint, test_dir (please add data of competition), mask_dir
