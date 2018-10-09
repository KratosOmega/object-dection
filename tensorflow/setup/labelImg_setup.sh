########################################################
#labelImg:
cd ../../../labelImg
sudo apt-get install pyqt5-dev-tools
#sudo apt-get install python3-setuptools
#sudo pip3 install --user pyqt5
sudo pip3 install -r requirements/requirements-linux-python3.txt
make qt5py3
#python3 labelImg.py [IMAGE_PATH] [PRE-DEFINED CLASS FILE]
#-------------------------------------------------------
