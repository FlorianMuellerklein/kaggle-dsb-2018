#/bin/bash

CUDA_VISIBLE_DEVICES="" python /home/florian/models/research/object_detection/eval.py --checkpoint_dir="/media/florian/Neumann/Kaggle/Data_Science_Bowl_2018/nuclei_detection/training" --pipeline_config_path="/media/florian/Neumann/Kaggle/Data_Science_Bowl_2018/nuclei_detection/faster_rcnn_resnet50_coco_2017_11_08/pipeline.config" --eval_dir="/media/florian/Neumann/Kaggle/Data_Science_Bowl_2018/nuclei_detection/eval" --logtostderr

