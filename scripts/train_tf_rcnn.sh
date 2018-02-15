#/bin/bash

CUDA_VISIBLE_DEVICES=1 python /home/florian/models/research/object_detection/train.py --train_dir="/media/florian/Neumann/Kaggle/Data_Science_Bowl_2018/nuclei_detection/training" --pipeline_config_path="/media/florian/Neumann/Kaggle/Data_Science_Bowl_2018/nuclei_detection/faster_rcnn_resnet50_coco_2017_11_08/pipeline.config" --logtostderr


