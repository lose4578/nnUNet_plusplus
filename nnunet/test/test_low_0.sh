# conda activate nnU_net
export CUDA_VISIBLE_DEVICES=0
python ../inference/predict_simple.py -i /media/dell/dell/seg/nnUNet/data/base/nnUNet_raw_splitted/Task00_KITS19/imagesTs -o /media/dell/dell/seg/nnUNet/test/all -t Task00_KITS19 -tr nnUNetTrainer -m 3d_lowres  -f 6 --part_id 0 --num_parts 3
