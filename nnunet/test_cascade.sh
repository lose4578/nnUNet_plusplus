python inference/predict_simple.py -i ../data/base/nnUNet_raw_splitted/Task00_KITS19/imagesTs/ \
-o ~/CV/nnUNet/test/3d_cascade/4 \
-t Task00_KITS19 -tr nnUNetTrainerCascadeFullRes -m 3d_cascade_fullres \
-f 4  -l /home/wangchengcheng/CV/nnUNet/test/3d_lowres/4
