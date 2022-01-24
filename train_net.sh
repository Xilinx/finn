#!/bin/bash
start_date=$(date)

# set personal environment
source /srv/cdl-eml/User/atchelet/pipenv_at/bin/activate
echo "pip environment activated"

export IMG_DIR=/srv/cdl-eml/User/atchelet/Dataset/images
export LBL_DIR=/srv/cdl-eml/User/atchelet/Dataset/labels

# script
cd /home/atchelet/git/finn_at/finn_at/finn_at/
echo "at $PWD ready to start"
time python3 train_net.py $IMG_DIR $LBL_DIR 4 4 200 64
echo "finished!"

end_date=$(date)

echo "Start"
echo $start_date
echo "End"
echo $end_date

#Command for executing the task
#ts -L qYOLO_train /home/atchelet/git/finn_at/finn_at/finn_at/train_net.sh