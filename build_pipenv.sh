#!/bin/bash
# move and set personal environment
cd /srv/cdl-eml/User/atchelet
python3 -m venv pipenv_at
cd pipenv_at
source bin/activate
echo "pip environment activated"

# install needed packages
pip3 install pip --upgrade
pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip3 install brevitas
pip3 install kmeans-pytorch
pip3 install -U scikit-image
pip3 install -U scikit-learn
pip3 install tqdm
pip3 install tensorboard


#Command for executing the task
#ts -L qYOLO_train /home/atchelet/git/finn_at/finn_at/finn_at/train_net.sh