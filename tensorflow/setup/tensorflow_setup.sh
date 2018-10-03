pip install tensorflow

#################################################################
# tensorflow dependencies
#################################################################
#echo "deb http://archive.ubuntu.com/ubuntu/ vivid universe" | sudo tee -a "/etc/apt/sources.list"
sudo apt-get install protobuf-compiler python3-pil python3-lxml python-tk
pip install --user Cython
pip install --user contextlib2
pip install --user jupyter
pip install --user matplotlib

#################################################################
# Add Libraries to PYTHONPATH
# Note: if .sh failed, then manually cd to
# tensorflow/models/research/
# and run export command as below
# also note that [pwd] should be updated with your correspounding
# pwd path
#################################################################
cd ../models/research/
#export PYTHONPATH=$PYTHONPATH:[pwd]:[pwd]/slim
export PYTHONPATH=$PYTHONPATH:~/Documents/object-detection/models/research:~/Documents/object-detection/models/research/slim




