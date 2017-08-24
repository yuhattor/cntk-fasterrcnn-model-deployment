# Deploy-CNTK-FasterRCNN
Please use this program to deploy Faster-R-CNN model trained by CNTK.
This is written in Python by using Flask.
I've tested only an AlexNet based model trained via python.
As you can see, this program is just a sample. If you want to try other models, I hope it works as well. But if it doesn't work, please let me know.
I hope this program helps someone who want to deploy an object detection solution trained by Faster-R-CNN algorithm.

Yuki


# Requirement
Minimum CNTK version required is 2.1

If you still use version 2.0 or older, please check these sites.

- For Windows user : https://docs.microsoft.com/en-us/cognitive-toolkit/Setup-Windows-Python
- For Linux user : https://docs.microsoft.com/en-us/cognitive-toolkit/setup-linux-python

For instance, the upgrade script is like below. (this is for Linux user with python 3.5)

`
pip install --upgrade --no-deps https://cntk.ai/PythonWheel/CPU-Only/cntk-2.1-cp35-cp35m-linux_x86_64.whl
`

# Training
This program is only for deployment. If you have to train models, please take a look at this repository.
https://github.com/Microsoft/CNTK/tree/master/Examples/Image/Detection/FasterRCNN


# How to use
To deploy this program is easy. Just run below code.

`
python application.py
`

But before you run this program, please check below.
- Your model file is correctly placed at `/Deploy-CNTK-FasterRCNN/web_deploy/faster.model`
- your labels are defined at `Deploy-CNTK-FasterRCNN/web_deploy/class_map.txt`
