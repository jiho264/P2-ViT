# CUDA_VISIBLE_DEVICES=1 python test_quant.py vit_base /home/shared_data/imagenet --quant --ptf --lis --quant-method minmax --mode 1
# # CUDA_VISIBLE_DEVICES=0 python test_quant.py vit_base /home/shared_data/imagenet --quant --quant-method minmax --mode 2
# gpu=(2 1)
# mode=(0 2)
# for i in 0
# do
# CUDA_VISIBLE_DEVICES=${gpu[$i]} nohup python -u test_quant.py vit_base /data/imagenet --quant --ptf --lis --quant-method minmax --mode ${mode[$i]} > logs/vit_base_48_4 2>&1 &
# done
CUDA_VISIBLE_DEVICES=3 python test_quant.py deit_small /data/imagenet --quant --ptf --lis --quant-method minmax --mode 0