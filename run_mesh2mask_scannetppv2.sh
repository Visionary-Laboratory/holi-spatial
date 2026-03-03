python PGSR/mesh2mask.py -m output_scannet/scannetppv2/027cd6ea0f -s scannetppv2/data/027cd6ea0f/dslr/nerfstudio/ --mesh_path /mnt/shared-storage-user/solution-gpfs02/liuyifei/pgsr_scannetppv2_all/027cd6ea0f/mesh/tsdf_fusion_post.ply -i scannetppv2/data/027cd6ea0f/dslr/resized_undistorted_images

python PGSR/mesh2mask.py -m output_scannet/scannetppv2/09d6e808b4 -s scannetppv2/data/09d6e808b4/dslr/nerfstudio/ --mesh_path /mnt/shared-storage-user/solution-gpfs02/liuyifei/pgsr_scannetppv2_all/09d6e808b4/mesh/tsdf_fusion_post.ply -i scannetppv2/data/09d6e808b4/dslr/resized_undistorted_images

python PGSR/mesh2mask.py -m output_scannet/scannetppv2/0a7cc12c0e -s scannetppv2/data/0a7cc12c0e/dslr/nerfstudio/ --mesh_path /mnt/shared-storage-user/solution-gpfs02/liuyifei/pgsr_scannetppv2_all/0a7cc12c0e/mesh/tsdf_fusion_post.ply -i scannetppv2/data/0a7cc12c0e/dslr/resized_undistorted_images

python PGSR/mesh2mask.py -m output_scannet/scannetppv2/0d8ead0038 -s scannetppv2/data/0d8ead0038/dslr/nerfstudio/ --mesh_path /mnt/shared-storage-user/solution-gpfs02/liuyifei/pgsr_scannetppv2_all/0d8ead0038/mesh/tsdf_fusion_post.ply -i scannetppv2/data/0d8ead0038/dslr/resized_undistorted_images

python PGSR/mesh2mask.py -m output_scannet/scannetppv2/116456116b -s scannetppv2/data/116456116b/dslr/nerfstudio/ --mesh_path /mnt/shared-storage-user/solution-gpfs02/liuyifei/pgsr_scannetppv2_all/116456116b/mesh/tsdf_fusion_post.ply -i scannetppv2/data/116456116b/dslr/resized_undistorted_images

python PGSR/mesh2mask.py -m output_scannet/scannetppv2/17a5e7d36c -s scannetppv2/data/17a5e7d36c/dslr/nerfstudio/ --mesh_path /mnt/shared-storage-user/solution-gpfs02/liuyifei/pgsr_scannetppv2_all/17a5e7d36c/mesh/tsdf_fusion_post.ply -i scannetppv2/data/17a5e7d36c/dslr/resized_undistorted_images

python PGSR/mesh2mask.py -m output_scannet/scannetppv2/1cefb55d50 -s scannetppv2/data/1cefb55d50/dslr/nerfstudio/ --mesh_path /mnt/shared-storage-user/solution-gpfs02/liuyifei/pgsr_scannetppv2_all/1cefb55d50/mesh/tsdf_fusion_post.ply -i scannetppv2/data/1cefb55d50/dslr/resized_undistorted_images

python PGSR/mesh2mask.py -m output_scannet/scannetppv2/20871b98f3 -s scannetppv2/data/20871b98f3/dslr/nerfstudio/ --mesh_path /mnt/shared-storage-user/solution-gpfs02/liuyifei/pgsr_scannetppv2_all/20871b98f3/mesh/tsdf_fusion_post.ply -i scannetppv2/data/20871b98f3/dslr/resized_undistorted_images

python PGSR/mesh2mask.py -m output_scannet/scannetppv2/924b364b9f -s scannetppv2/data/924b364b9f/dslr/nerfstudio/ --mesh_path /mnt/shared-storage-user/solution-gpfs02/liuyifei/pgsr_scannetppv2_all/924b364b9f/mesh/tsdf_fusion_post.ply -i scannetppv2/data/924b364b9f/dslr/resized_undistorted_images

python PGSR/mesh2mask.py -m output_scannet/scannetppv2/00a231a370 -s scannetppv2/data/00a231a370/dslr/nerfstudio/ --mesh_path /mnt/shared-storage-user/solution-gpfs02/liuyifei/pgsr_scannetppv2_all/00a231a370/mesh/tsdf_fusion_post.ply -i scannetppv2/data/00a231a370/dslr/resized_undistorted_images


# python 3d_bounding_instance_gs_rerun.py --scene 00a231a370 --data-root scannetppv2/data/ --mask-root /mnt/shared-storage-user/solution/liuyifei/code/posevlm/sam_masks_debug_scannet_old/ -m output_scannet/scannetppv2/00a231a370/ --output-dir output_3d_bounding_scannet_evaluation_new --vllm-api-url http://100.102.160.232:8000/v1/chat/completions

python 3d_bounding_instance_gs_rerun.py --scene 027cd6ea0f --data-root scannetppv2/data/ --mask-root /mnt/shared-storage-user/solution/liuyifei/code/posevlm/sam_masks_debug_scannet_old/ -m output_scannet/scannetppv2/027cd6ea0f/ --output-dir output_3d_bounding_scannet_evaluation_new --vllm-api-url http://100.102.160.232:8000/v1/chat/completions

python 3d_bounding_instance_gs_rerun.py --scene 09d6e808b4 --data-root scannetppv2/data/ --mask-root /mnt/shared-storage-user/solution/liuyifei/code/posevlm/sam_masks_debug_scannet_old/ -m output_scannet/scannetppv2/09d6e808b4/ --output-dir output_3d_bounding_scannet_evaluation_new --vllm-api-url http://100.102.160.232:8000/v1/chat/completions

python 3d_bounding_instance_gs_rerun.py --scene 0a7cc12c0e --data-root scannetppv2/data/ --mask-root /mnt/shared-storage-user/solution/liuyifei/code/posevlm/sam_masks_debug_scannet_old/ -m output_scannet/scannetppv2/0a7cc12c0e/ --output-dir output_3d_bounding_scannet_evaluation_new --vllm-api-url http://100.102.160.232:8000/v1/chat/completions

python 3d_bounding_instance_gs_rerun.py --scene 0d8ead0038 --data-root scannetppv2/data/ --mask-root /mnt/shared-storage-user/solution/liuyifei/code/posevlm/sam_masks_debug_scannet_old/ -m output_scannet/scannetppv2/0d8ead0038/ --output-dir output_3d_bounding_scannet_evaluation_new --vllm-api-url http://100.102.160.232:8000/v1/chat/completions

python 3d_bounding_instance_gs_rerun.py --scene 116456116b --data-root scannetppv2/data/ --mask-root /mnt/shared-storage-user/solution/liuyifei/code/posevlm/sam_masks_debug_scannet_old/ -m output_scannet/scannetppv2/116456116b/ --output-dir output_3d_bounding_scannet_evaluation_new --vllm-api-url http://100.102.160.232:8000/v1/chat/completions

python 3d_bounding_instance_gs_rerun.py --scene 17a5e7d36c --data-root scannetppv2/data/ --mask-root /mnt/shared-storage-user/solution/liuyifei/code/posevlm/sam_masks_debug_scannet_old/ -m output_scannet/scannetppv2/17a5e7d36c/ --output-dir output_3d_bounding_scannet_evaluation_new --vllm-api-url http://100.102.160.232:8000/v1/chat/completions

python 3d_bounding_instance_gs_rerun.py --scene 1cefb55d50 --data-root scannetppv2/data/ --mask-root /mnt/shared-storage-user/solution/liuyifei/code/posevlm/sam_masks_debug_scannet_old/ -m output_scannet/scannetppv2/1cefb55d50/ --output-dir output_3d_bounding_scannet_evaluation_new --vllm-api-url http://100.102.160.232:8000/v1/chat/completions

python 3d_bounding_instance_gs_rerun.py --scene 20871b98f3 --data-root scannetppv2/data/ --mask-root /mnt/shared-storage-user/solution/liuyifei/code/posevlm/sam_masks_debug_scannet_old/ -m output_scannet/scannetppv2/20871b98f3/ --output-dir output_3d_bounding_scannet_evaluation_new --vllm-api-url http://100.102.160.232:8000/v1/chat/completions

python 3d_bounding_instance_gs_rerun.py --scene 924b364b9f --data-root scannetppv2/data/ --mask-root /mnt/shared-storage-user/solution/liuyifei/code/posevlm/sam_masks_debug_scannet_old/ -m output_scannet/scannetppv2/924b364b9f/ --output-dir output_3d_bounding_scannet_evaluation_new --vllm-api-url http://100.102.160.232:8000/v1/chat/completions


# ========== 3d_bounding_instance_gs_rerun_wo_confidence.py 命令 ==========

python 3d_bounding_instance_gs_rerun_wo_confidence.py --scene 027cd6ea0f --data-root scannetppv2/data/ --mask-root /mnt/shared-storage-user/solution/liuyifei/code/posevlm/sam_masks_debug_scannet_old/ -m output_scannet/scannetppv2/027cd6ea0f/ --output-dir output_3d_bounding_scannet_evaluation_woconf --vllm-api-url http://100.102.160.232:8000/v1/chat/completions

python 3d_bounding_instance_gs_rerun_wo_confidence.py --scene 09d6e808b4 --data-root scannetppv2/data/ --mask-root /mnt/shared-storage-user/solution/liuyifei/code/posevlm/sam_masks_debug_scannet_old/ -m output_scannet/scannetppv2/09d6e808b4/ --output-dir output_3d_bounding_scannet_evaluation_woconf --vllm-api-url http://100.102.160.232:8000/v1/chat/completions

python 3d_bounding_instance_gs_rerun_wo_confidence.py --scene 0a7cc12c0e --data-root scannetppv2/data/ --mask-root /mnt/shared-storage-user/solution/liuyifei/code/posevlm/sam_masks_debug_scannet_old/ -m output_scannet/scannetppv2/0a7cc12c0e/ --output-dir output_3d_bounding_scannet_evaluation_woconf --vllm-api-url http://100.102.160.232:8000/v1/chat/completions

python 3d_bounding_instance_gs_rerun_wo_confidence.py --scene 0d8ead0038 --data-root scannetppv2/data/ --mask-root /mnt/shared-storage-user/solution/liuyifei/code/posevlm/sam_masks_debug_scannet_old/ -m output_scannet/scannetppv2/0d8ead0038/ --output-dir output_3d_bounding_scannet_evaluation_woconf --vllm-api-url http://100.102.160.232:8000/v1/chat/completions

python 3d_bounding_instance_gs_rerun_wo_confidence.py --scene 116456116b --data-root scannetppv2/data/ --mask-root /mnt/shared-storage-user/solution/liuyifei/code/posevlm/sam_masks_debug_scannet_old/ -m output_scannet/scannetppv2/116456116b/ --output-dir output_3d_bounding_scannet_evaluation_woconf --vllm-api-url http://100.102.160.232:8000/v1/chat/completions

python 3d_bounding_instance_gs_rerun_wo_confidence.py --scene 17a5e7d36c --data-root scannetppv2/data/ --mask-root /mnt/shared-storage-user/solution/liuyifei/code/posevlm/sam_masks_debug_scannet_old/ -m output_scannet/scannetppv2/17a5e7d36c/ --output-dir output_3d_bounding_scannet_evaluation_woconf --vllm-api-url http://100.102.160.232:8000/v1/chat/completions

python 3d_bounding_instance_gs_rerun_wo_confidence.py --scene 1cefb55d50 --data-root scannetppv2/data/ --mask-root /mnt/shared-storage-user/solution/liuyifei/code/posevlm/sam_masks_debug_scannet_old/ -m output_scannet/scannetppv2/1cefb55d50/ --output-dir output_3d_bounding_scannet_evaluation_woconf --vllm-api-url http://100.102.160.232:8000/v1/chat/completions

python 3d_bounding_instance_gs_rerun_wo_confidence.py --scene 20871b98f3 --data-root scannetppv2/data/ --mask-root /mnt/shared-storage-user/solution/liuyifei/code/posevlm/sam_masks_debug_scannet_old/ -m output_scannet/scannetppv2/20871b98f3/ --output-dir output_3d_bounding_scannet_evaluation_woconf --vllm-api-url http://100.102.160.232:8000/v1/chat/completions

python 3d_bounding_instance_gs_rerun_wo_confidence.py --scene 924b364b9f --data-root scannetppv2/data/ --mask-root /mnt/shared-storage-user/solution/liuyifei/code/posevlm/sam_masks_debug_scannet_old/ -m output_scannet/scannetppv2/924b364b9f/ --output-dir output_3d_bounding_scannet_evaluation_woconf --vllm-api-url http://100.102.160.232:8000/v1/chat/completions


# ========== 3d_bounding_instance_gs_rerun_wovlm.py 命令 ==========

python 3d_bounding_instance_gs_rerun_wovlm.py --scene 027cd6ea0f --data-root scannetppv2/data/ --mask-root /mnt/shared-storage-user/solution/liuyifei/code/posevlm/sam_masks_debug_scannet_old/ -m output_scannet/scannetppv2/027cd6ea0f/ --output-dir output_3d_bounding_scannet_evaluation_wovlm --vllm-api-url http://100.102.160.232:8000/v1/chat/completions

python 3d_bounding_instance_gs_rerun_wovlm.py --scene 09d6e808b4 --data-root scannetppv2/data/ --mask-root /mnt/shared-storage-user/solution/liuyifei/code/posevlm/sam_masks_debug_scannet_old/ -m output_scannet/scannetppv2/09d6e808b4/ --output-dir output_3d_bounding_scannet_evaluation_wovlm --vllm-api-url http://100.102.160.232:8000/v1/chat/completions

python 3d_bounding_instance_gs_rerun_wovlm.py --scene 0a7cc12c0e --data-root scannetppv2/data/ --mask-root /mnt/shared-storage-user/solution/liuyifei/code/posevlm/sam_masks_debug_scannet_old/ -m output_scannet/scannetppv2/0a7cc12c0e/ --output-dir output_3d_bounding_scannet_evaluation_wovlm --vllm-api-url http://100.102.160.232:8000/v1/chat/completions

python 3d_bounding_instance_gs_rerun_wovlm.py --scene 0d8ead0038 --data-root scannetppv2/data/ --mask-root /mnt/shared-storage-user/solution/liuyifei/code/posevlm/sam_masks_debug_scannet_old/ -m output_scannet/scannetppv2/0d8ead0038/ --output-dir output_3d_bounding_scannet_evaluation_wovlm --vllm-api-url http://100.102.160.232:8000/v1/chat/completions

python 3d_bounding_instance_gs_rerun_wovlm.py --scene 116456116b --data-root scannetppv2/data/ --mask-root /mnt/shared-storage-user/solution/liuyifei/code/posevlm/sam_masks_debug_scannet_old/ -m output_scannet/scannetppv2/116456116b/ --output-dir output_3d_bounding_scannet_evaluation_wovlm --vllm-api-url http://100.102.160.232:8000/v1/chat/completions

python 3d_bounding_instance_gs_rerun_wovlm.py --scene 17a5e7d36c --data-root scannetppv2/data/ --mask-root /mnt/shared-storage-user/solution/liuyifei/code/posevlm/sam_masks_debug_scannet_old/ -m output_scannet/scannetppv2/17a5e7d36c/ --output-dir output_3d_bounding_scannet_evaluation_wovlm --vllm-api-url http://100.102.160.232:8000/v1/chat/completions

python 3d_bounding_instance_gs_rerun_wovlm.py --scene 1cefb55d50 --data-root scannetppv2/data/ --mask-root /mnt/shared-storage-user/solution/liuyifei/code/posevlm/sam_masks_debug_scannet_old/ -m output_scannet/scannetppv2/1cefb55d50/ --output-dir output_3d_bounding_scannet_evaluation_wovlm --vllm-api-url http://100.102.160.232:8000/v1/chat/completions

python 3d_bounding_instance_gs_rerun_wovlm.py --scene 20871b98f3 --data-root scannetppv2/data/ --mask-root /mnt/shared-storage-user/solution/liuyifei/code/posevlm/sam_masks_debug_scannet_old/ -m output_scannet/scannetppv2/20871b98f3/ --output-dir output_3d_bounding_scannet_evaluation_wovlm --vllm-api-url http://100.102.160.232:8000/v1/chat/completions

python 3d_bounding_instance_gs_rerun_wovlm.py --scene 924b364b9f --data-root scannetppv2/data/ --mask-root /mnt/shared-storage-user/solution/liuyifei/code/posevlm/sam_masks_debug_scannet_old/ -m output_scannet/scannetppv2/924b364b9f/ --output-dir output_3d_bounding_scannet_evaluation_wovlm --vllm-api-url http://100.102.160.232:8000/v1/chat/completions


# ========== 3d_bounding_instance_gs_rerun_da3.py 命令 ==========

python 3d_bounding_instance_gs_rerun_da3.py --scene 027cd6ea0f --data-root scannetppv2/data/ --mask-root /mnt/shared-storage-user/solution/liuyifei/code/posevlm/sam_masks_debug_scannet_old/ -m output_scannet/scannetppv2/027cd6ea0f/ --output-dir output_3d_bounding_scannet_evaluation_da3 --vllm-api-url http://100.102.160.232:8000/v1/chat/completions

python 3d_bounding_instance_gs_rerun_da3.py --scene 09d6e808b4 --data-root scannetppv2/data/ --mask-root /mnt/shared-storage-user/solution/liuyifei/code/posevlm/sam_masks_debug_scannet_old/ -m output_scannet/scannetppv2/09d6e808b4/ --output-dir output_3d_bounding_scannet_evaluation_da3 --vllm-api-url http://100.102.160.232:8000/v1/chat/completions

python 3d_bounding_instance_gs_rerun_da3.py --scene 0a7cc12c0e --data-root scannetppv2/data/ --mask-root /mnt/shared-storage-user/solution/liuyifei/code/posevlm/sam_masks_debug_scannet_old/ -m output_scannet/scannetppv2/0a7cc12c0e/ --output-dir output_3d_bounding_scannet_evaluation_da3 --vllm-api-url http://100.102.160.232:8000/v1/chat/completions

python 3d_bounding_instance_gs_rerun_da3.py --scene 0d8ead0038 --data-root scannetppv2/data/ --mask-root /mnt/shared-storage-user/solution/liuyifei/code/posevlm/sam_masks_debug_scannet_old/ -m output_scannet/scannetppv2/0d8ead0038/ --output-dir output_3d_bounding_scannet_evaluation_da3 --vllm-api-url http://100.102.160.232:8000/v1/chat/completions

python 3d_bounding_instance_gs_rerun_da3.py --scene 116456116b --data-root scannetppv2/data/ --mask-root /mnt/shared-storage-user/solution/liuyifei/code/posevlm/sam_masks_debug_scannet_old/ -m output_scannet/scannetppv2/116456116b/ --output-dir output_3d_bounding_scannet_evaluation_da3 --vllm-api-url http://100.102.160.232:8000/v1/chat/completions

python 3d_bounding_instance_gs_rerun_da3.py --scene 17a5e7d36c --data-root scannetppv2/data/ --mask-root /mnt/shared-storage-user/solution/liuyifei/code/posevlm/sam_masks_debug_scannet_old/ -m output_scannet/scannetppv2/17a5e7d36c/ --output-dir output_3d_bounding_scannet_evaluation_da3 --vllm-api-url http://100.102.160.232:8000/v1/chat/completions

python 3d_bounding_instance_gs_rerun_da3.py --scene 1cefb55d50 --data-root scannetppv2/data/ --mask-root /mnt/shared-storage-user/solution/liuyifei/code/posevlm/sam_masks_debug_scannet_old/ -m output_scannet/scannetppv2/1cefb55d50/ --output-dir output_3d_bounding_scannet_evaluation_da3 --vllm-api-url http://100.102.160.232:8000/v1/chat/completions

python 3d_bounding_instance_gs_rerun_da3.py --scene 20871b98f3 --data-root scannetppv2/data/ --mask-root /mnt/shared-storage-user/solution/liuyifei/code/posevlm/sam_masks_debug_scannet_old/ -m output_scannet/scannetppv2/20871b98f3/ --output-dir output_3d_bounding_scannet_evaluation_da3 --vllm-api-url http://100.102.160.232:8000/v1/chat/completions

python 3d_bounding_instance_gs_rerun_da3.py --scene 924b364b9f --data-root scannetppv2/data/ --mask-root /mnt/shared-storage-user/solution/liuyifei/code/posevlm/sam_masks_debug_scannet_old/ -m output_scannet/scannetppv2/924b364b9f/ --output-dir output_3d_bounding_scannet_evaluation_da3 --vllm-api-url http://100.102.160.232:8000/v1/chat/completions


# ========== 3d_bounding_instance_gs_rerun_da3.py 命令 ==========

python 3d_bounding_instance_gs_rerun_da3.py --scene 027cd6ea0f --data-root scannetppv2/data/ --mask-root /mnt/shared-storage-user/solution/liuyifei/code/posevlm/sam_masks_debug_scannet_old/ -m output_scannet/scannetppv2/027cd6ea0f/ --output-dir output_3d_bounding_scannet_evaluation_da3 --vllm-api-url http://100.102.160.232:8000/v1/chat/completions

python 3d_bounding_instance_gs_rerun_da3.py --scene 09d6e808b4 --data-root scannetppv2/data/ --mask-root /mnt/shared-storage-user/solution/liuyifei/code/posevlm/sam_masks_debug_scannet_old/ -m output_scannet/scannetppv2/09d6e808b4/ --output-dir output_3d_bounding_scannet_evaluation_da3 --vllm-api-url http://100.102.160.232:8000/v1/chat/completions

python 3d_bounding_instance_gs_rerun_da3.py --scene 0a7cc12c0e --data-root scannetppv2/data/ --mask-root /mnt/shared-storage-user/solution/liuyifei/code/posevlm/sam_masks_debug_scannet_old/ -m output_scannet/scannetppv2/0a7cc12c0e/ --output-dir output_3d_bounding_scannet_evaluation_da3 --vllm-api-url http://100.102.160.232:8000/v1/chat/completions

python 3d_bounding_instance_gs_rerun_da3.py --scene 0d8ead0038 --data-root scannetppv2/data/ --mask-root /mnt/shared-storage-user/solution/liuyifei/code/posevlm/sam_masks_debug_scannet_old/ -m output_scannet/scannetppv2/0d8ead0038/ --output-dir output_3d_bounding_scannet_evaluation_da3 --vllm-api-url http://100.102.160.232:8000/v1/chat/completions

python 3d_bounding_instance_gs_rerun_da3.py --scene 116456116b --data-root scannetppv2/data/ --mask-root /mnt/shared-storage-user/solution/liuyifei/code/posevlm/sam_masks_debug_scannet_old/ -m output_scannet/scannetppv2/116456116b/ --output-dir output_3d_bounding_scannet_evaluation_da3 --vllm-api-url http://100.102.160.232:8000/v1/chat/completions

python 3d_bounding_instance_gs_rerun_da3.py --scene 17a5e7d36c --data-root scannetppv2/data/ --mask-root /mnt/shared-storage-user/solution/liuyifei/code/posevlm/sam_masks_debug_scannet_old/ -m output_scannet/scannetppv2/17a5e7d36c/ --output-dir output_3d_bounding_scannet_evaluation_da3 --vllm-api-url http://100.102.160.232:8000/v1/chat/completions

python 3d_bounding_instance_gs_rerun_da3.py --scene 1cefb55d50 --data-root scannetppv2/data/ --mask-root /mnt/shared-storage-user/solution/liuyifei/code/posevlm/sam_masks_debug_scannet_old/ -m output_scannet/scannetppv2/1cefb55d50/ --output-dir output_3d_bounding_scannet_evaluation_da3 --vllm-api-url http://100.102.160.232:8000/v1/chat/completions

python 3d_bounding_instance_gs_rerun_da3.py --scene 20871b98f3 --data-root scannetppv2/data/ --mask-root /mnt/shared-storage-user/solution/liuyifei/code/posevlm/sam_masks_debug_scannet_old/ -m output_scannet/scannetppv2/20871b98f3/ --output-dir output_3d_bounding_scannet_evaluation_da3 --vllm-api-url http://100.102.160.232:8000/v1/chat/completions

python 3d_bounding_instance_gs_rerun_da3.py --scene 924b364b9f --data-root scannetppv2/data/ --mask-root /mnt/shared-storage-user/solution/liuyifei/code/posevlm/sam_masks_debug_scannet_old/ -m output_scannet/scannetppv2/924b364b9f/ --output-dir output_3d_bounding_scannet_evaluation_da3 --vllm-api-url http://100.102.160.232:8000/v1/chat/completions



python 3d_bounding_instance_gs_rerun_da3.py --scene 924b364b9f --data-root scannetppv2/data/ --mask-root /mnt/shared-storage-user/solution/liuyifei/code/posevlm/sam_masks_debug_scannet_old/ -m output_scannet/scannetppv2/924b364b9f/ --output-dir output_3d_bounding_scannet_evaluation_da3 --vllm-api-url http://100.102.160.232:8000/v1/chat/completions



python depth2ply.py \
  --scene 0a7cc12c0e \
  --data-root scannetppv2/data \
  -m output_scannet/scannetppv2/0a7cc12c0e/ \
  --iteration 30000 \
  --output output_points.ply