export CUDA_VISIBLE_DEVICES=0
python ../inference/predict_simple.py -i ../../data/base/nnUNet_raw_splitted/Task00_KITS19/imagesTs/ -o /media/dell/dell/seg/nnUNet/test/3d_cascade/4 -t Task00_KITS19 -tr nnUNetTrainerCascadeFullRes -m 3d_cascade_fullres -f 4 --part_id 0 --num_parts 3 -l /media/dell/dell/seg/nnUNet/test/3d_lowres/4/

