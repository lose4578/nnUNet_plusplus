# export CUDA_VISIBLE_DEVICES=0
python ../inference/predict_simple.py -i /media/dell/dell/seg/nnUNet/final_test/img -o /media/dell/dell/seg/nnUNet/final_test/lable_test/final_3  -t Task00_KITS19 -tr nnUNetTrainer -m 3d_lowres  -f 0
