# 8 卡并行，每卡处理一份场景列表（在项目根目录下运行）
CUDA_VISIBLE_DEVICES=0 python inference_da3_scannetppv2.py --scene_list_txt scannetv2/splits/scenes_part1.txt &
CUDA_VISIBLE_DEVICES=1 python inference_da3_scannetppv2.py --scene_list_txt scannetv2/splits/scenes_part2.txt &
CUDA_VISIBLE_DEVICES=2 python inference_da3_scannetppv2.py --scene_list_txt scannetv2/splits/scenes_part3.txt &
CUDA_VISIBLE_DEVICES=3 python inference_da3_scannetppv2.py --scene_list_txt scannetv2/splits/scenes_part4.txt &
CUDA_VISIBLE_DEVICES=4 python inference_da3_scannetppv2.py --scene_list_txt scannetv2/splits/scenes_part5.txt &
CUDA_VISIBLE_DEVICES=5 python inference_da3_scannetppv2.py --scene_list_txt scannetv2/splits/scenes_part6.txt &
CUDA_VISIBLE_DEVICES=6 python inference_da3_scannetppv2.py --scene_list_txt scannetv2/splits/scenes_part7.txt &
CUDA_VISIBLE_DEVICES=7 python inference_da3_scannetppv2.py --scene_list_txt scannetv2/splits/scenes_part8.txt &
wait
echo "All 8 jobs finished."