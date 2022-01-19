#!/bin/bash
start_date=$(date)

# move and set personal environment
cd /srv/cdl-eml/User/atchelet
python3 -m venv pipenv_at
cd pipenv_at
source bin/activate

# install needed packages
pip3 install pip --upgrade
pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip3 install brevitas
pip3 install tensorboard
pip3 install -U scikit-image
pip3 install -U scikit-learn

export IMG_DIR=/srv/cdl-eml/User/atchelet/dataset/Dataset/images
export LBL_DIR=/srv/cdl-eml/User/atchelet/dataset/Dataset/labels

# script
time python3 train_net.py $IMG_DIR $LBL_DIR 1 3 150

end_date=$(date)

echo "Start"
echo $start_date
echo "End"
echo $end_date