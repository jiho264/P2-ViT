# CUDA_VISIBLE_DEVICES=1 python test_quant.py vit_base /home/shared_data/imagenet --quant --ptf --lis --quant-method minmax --mode 1
# # CUDA_VISIBLE_DEVICES=0 python test_quant.py vit_base /home/shared_data/imagenet --quant --quant-method minmax --mode 2
gpu=(0 2 3)
mode=(0 1 2)
for i in 0 1 2
do
CUDA_VISIBLE_DEVICES=${gpu[$i]} nohup python -u test_quant.py vit_base /mnt/HD_1/datasets/imagenet --quant --ptf --lis --quant-method minmax --mode ${mode[$i]} > logs/vit_base_${mode[$i]}_48 2>&1 &
done
# CUDA_VISIBLE_DEVICES=0 python test_quant.py vit_base /home/shared_data/imagenet --quant --quant-method minmax --mode 2