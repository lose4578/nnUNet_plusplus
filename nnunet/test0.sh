CUDA_VISIBLE_DEVICES=0
python inference/predict_simple.py -i ../data/base/nnUNet_raw_splitted/Task00_KITS19/imagesTs -o /media/dell/dell/seg/nnUNet/test -t Task00_KITS19 -tr nnUNetTrainer -m 3d_fullres --part_id=0 --num_parts=3 -f 4

