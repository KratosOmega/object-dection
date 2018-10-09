##############################################################
# Convert xml to csv
##############################################################
#python3 ./prepare_dataset/xml_to_csv.py
#-------------------------------------------------------------



##############################################################
# Convert csv to tfRecord
# train & test should run separately
# & enable and disable corresponding code block in
# ./prepare_dataset/generate_tfrecord.py for train & test
##############################################################

# ------------------------------------------------------ train
#python3 ./xml_to_csv.py
#python3 ./generate_tfrecord.py --csv_input=../../../data/csv/auvsi_train_annotation.csv --output_path=../../../data/record/auvsi_train_annotation.record


# ------------------------------------------------------- test
#python3 ./xml_to_csv.py
#python3 ./generate_tfrecord.py --csv_input=../../../data/csv/auvsi_test_annotation.csv --output_path=../../../data/record/auvsi_test_annotation.record
#-------------------------------------------------------------









# cd models/research/object_detection/legacy/
# python3 train.py --logtostderr --train_dir=path/to/output/directory --pipeline_config_path=path/to/CONFIG.config
# python3 train.py --logtostderr --train_dir=./training --pipeline_config_path=./training/ssd_mobilenet_v1_pets.config

