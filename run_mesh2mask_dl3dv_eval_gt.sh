#!/bin/bash

# 场景1
# python PGSR/mesh2mask.py -m output_DL3DV/1K/0bfab69b5b2b692d066b4435f2f6406357772aa53a3e17c3e0b8598083848a7a/ -s DL3DV/1K/0bfab69b5b2b692d066b4435f2f6406357772aa53a3e17c3e0b8598083848a7a/ --mesh_path /mnt/shared-storage-user/solution-gpfs02/liuyifei/pgsr_DL3DV_all/1K/0bfab69b5b2b692d066b4435f2f6406357772aa53a3e17c3e0b8598083848a7a/mesh/tsdf_fusion_post.ply

# 场景2
python PGSR/mesh2mask.py -m output_DL3DV/1K/2d52e65f703cf307013273ffeea79b76db153d665294becb16f00477aac05a88/ -s DL3DV/1K/2d52e65f703cf307013273ffeea79b76db153d665294becb16f00477aac05a88/ --mesh_path /mnt/shared-storage-user/solution-gpfs02/liuyifei/pgsr_DL3DV_all/1K/2d52e65f703cf307013273ffeea79b76db153d665294becb16f00477aac05a88/mesh/tsdf_fusion_post.ply

# 场景3
python PGSR/mesh2mask.py -m output_DL3DV/1K/5b1e56386e263f2324365d2c623d6b746280aa4f5a6257bb0ba66e5dff135c8a/ -s DL3DV/1K/5b1e56386e263f2324365d2c623d6b746280aa4f5a6257bb0ba66e5dff135c8a/ --mesh_path /mnt/shared-storage-user/solution-gpfs02/liuyifei/pgsr_DL3DV_all/1K/5b1e56386e263f2324365d2c623d6b746280aa4f5a6257bb0ba66e5dff135c8a/mesh/tsdf_fusion_post.ply

# 场景4
python PGSR/mesh2mask.py -m output_DL3DV/1K/7ae35f524e63aa86d1985f1c59dfd7a29cbd6069770ee26506ac26705917d9ea/ -s DL3DV/1K/7ae35f524e63aa86d1985f1c59dfd7a29cbd6069770ee26506ac26705917d9ea/ --mesh_path /mnt/shared-storage-user/solution-gpfs02/liuyifei/pgsr_DL3DV_all/1K/7ae35f524e63aa86d1985f1c59dfd7a29cbd6069770ee26506ac26705917d9ea/mesh/tsdf_fusion_post.ply

# 场景5
python PGSR/mesh2mask.py -m output_DL3DV/1K/7ed574ca0081fa9a8c879a4b360ec7b274b3a9d248c1a6e66bd1e1d1c47bea4b/ -s DL3DV/1K/7ed574ca0081fa9a8c879a4b360ec7b274b3a9d248c1a6e66bd1e1d1c47bea4b/ --mesh_path /mnt/shared-storage-user/solution-gpfs02/liuyifei/pgsr_DL3DV_all/1K/7ed574ca0081fa9a8c879a4b360ec7b274b3a9d248c1a6e66bd1e1d1c47bea4b/mesh/tsdf_fusion_post.ply

# 场景6
python PGSR/mesh2mask.py -m output_DL3DV/1K/7f5f6eac88ce535147bb9a7740635901929af1b08fbf0dd1adf28bf8aeee380a/ -s DL3DV/1K/7f5f6eac88ce535147bb9a7740635901929af1b08fbf0dd1adf28bf8aeee380a/ --mesh_path /mnt/shared-storage-user/solution-gpfs02/liuyifei/pgsr_DL3DV_all/1K/7f5f6eac88ce535147bb9a7740635901929af1b08fbf0dd1adf28bf8aeee380a/mesh/tsdf_fusion_post.ply

# 场景7
python PGSR/mesh2mask.py -m output_DL3DV/1K/b3d3847b167ee770b105de4388108a1b8f3bfaa430d4d6ff0ba910d271d8a487/ -s DL3DV/1K/b3d3847b167ee770b105de4388108a1b8f3bfaa430d4d6ff0ba910d271d8a487/ --mesh_path /mnt/shared-storage-user/solution-gpfs02/liuyifei/pgsr_DL3DV_all/1K/b3d3847b167ee770b105de4388108a1b8f3bfaa430d4d6ff0ba910d271d8a487/mesh/tsdf_fusion_post.ply

# 场景8
python PGSR/mesh2mask.py -m output_DL3DV/1K/c07ac2cdc0dddc8fe07c42258725ca0a5b486dcaccaef4fcbe8f5311cdee46fc/ -s DL3DV/1K/c07ac2cdc0dddc8fe07c42258725ca0a5b486dcaccaef4fcbe8f5311cdee46fc/ --mesh_path /mnt/shared-storage-user/solution-gpfs02/liuyifei/pgsr_DL3DV_all/1K/c07ac2cdc0dddc8fe07c42258725ca0a5b486dcaccaef4fcbe8f5311cdee46fc/mesh/tsdf_fusion_post.ply

# 场景9
python PGSR/mesh2mask.py -m output_DL3DV/1K/cd6c19a00fe93297c128769f81f7778f8c2b7a6ca7d1ca780b3f83803cdc2082/ -s DL3DV/1K/cd6c19a00fe93297c128769f81f7778f8c2b7a6ca7d1ca780b3f83803cdc2082/ --mesh_path /mnt/shared-storage-user/solution-gpfs02/liuyifei/pgsr_DL3DV_all/1K/cd6c19a00fe93297c128769f81f7778f8c2b7a6ca7d1ca780b3f83803cdc2082/mesh/tsdf_fusion_post.ply

# ========== 3d_bounding_instance_gs_rerun.py 命令 ==========

# 场景1
python 3d_bounding_instance_gs_rerun.py --scene 0bfab69b5b2b692d066b4435f2f6406357772aa53a3e17c3e0b8598083848a7a --data-root DL3DV/1K/ -m output_DL3DV/1K/0bfab69b5b2b692d066b4435f2f6406357772aa53a3e17c3e0b8598083848a7a/ --output-dir output_3d_bounding_dl3dv_evaluation_new --vllm-api-url http://100.102.160.232:8000/v1/chat/completions --mask-root sam_masks_debug_dl3dv/

# 场景2
python 3d_bounding_instance_gs_rerun.py --scene 2d52e65f703cf307013273ffeea79b76db153d665294becb16f00477aac05a88 --data-root DL3DV/1K/ -m output_DL3DV/1K/2d52e65f703cf307013273ffeea79b76db153d665294becb16f00477aac05a88/ --output-dir output_3d_bounding_dl3dv_evaluation_new --vllm-api-url http://100.102.160.232:8000/v1/chat/completions --mask-root sam_masks_debug_dl3dv/

# 场景3
python 3d_bounding_instance_gs_rerun.py --scene 5b1e56386e263f2324365d2c623d6b746280aa4f5a6257bb0ba66e5dff135c8a --data-root DL3DV/1K/ -m output_DL3DV/1K/5b1e56386e263f2324365d2c623d6b746280aa4f5a6257bb0ba66e5dff135c8a/ --output-dir output_3d_bounding_dl3dv_evaluation_new --vllm-api-url http://100.102.160.232:8000/v1/chat/completions --mask-root sam_masks_debug_dl3dv/

# 场景4
python 3d_bounding_instance_gs_rerun.py --scene 7ae35f524e63aa86d1985f1c59dfd7a29cbd6069770ee26506ac26705917d9ea --data-root DL3DV/1K/ -m output_DL3DV/1K/7ae35f524e63aa86d1985f1c59dfd7a29cbd6069770ee26506ac26705917d9ea/ --output-dir output_3d_bounding_dl3dv_evaluation_new --vllm-api-url http://100.102.160.232:8000/v1/chat/completions --mask-root sam_masks_debug_dl3dv/

# 场景5
python 3d_bounding_instance_gs_rerun.py --scene 7ed574ca0081fa9a8c879a4b360ec7b274b3a9d248c1a6e66bd1e1d1c47bea4b --data-root DL3DV/1K/ -m output_DL3DV/1K/7ed574ca0081fa9a8c879a4b360ec7b274b3a9d248c1a6e66bd1e1d1c47bea4b/ --output-dir output_3d_bounding_dl3dv_evaluation_new --vllm-api-url http://100.102.160.232:8000/v1/chat/completions --mask-root sam_masks_debug_dl3dv/

# 场景6
python 3d_bounding_instance_gs_rerun.py --scene 7f5f6eac88ce535147bb9a7740635901929af1b08fbf0dd1adf28bf8aeee380a --data-root DL3DV/1K/ -m output_DL3DV/1K/7f5f6eac88ce535147bb9a7740635901929af1b08fbf0dd1adf28bf8aeee380a/ --output-dir output_3d_bounding_dl3dv_evaluation_new --vllm-api-url http://100.102.160.232:8000/v1/chat/completions --mask-root sam_masks_debug_dl3dv/

# 场景7
python 3d_bounding_instance_gs_rerun.py --scene b3d3847b167ee770b105de4388108a1b8f3bfaa430d4d6ff0ba910d271d8a487 --data-root DL3DV/1K/ -m output_DL3DV/1K/b3d3847b167ee770b105de4388108a1b8f3bfaa430d4d6ff0ba910d271d8a487/ --output-dir output_3d_bounding_dl3dv_evaluation_new --vllm-api-url http://100.102.160.232:8000/v1/chat/completions --mask-root sam_masks_debug_dl3dv/

# 场景8
python 3d_bounding_instance_gs_rerun.py --scene c07ac2cdc0dddc8fe07c42258725ca0a5b486dcaccaef4fcbe8f5311cdee46fc --data-root DL3DV/1K/ -m output_DL3DV/1K/c07ac2cdc0dddc8fe07c42258725ca0a5b486dcaccaef4fcbe8f5311cdee46fc/ --output-dir output_3d_bounding_dl3dv_evaluation_new --vllm-api-url http://100.102.160.232:8000/v1/chat/completions --mask-root sam_masks_debug_dl3dv/

# 场景9
python 3d_bounding_instance_gs_rerun.py --scene cd6c19a00fe93297c128769f81f7778f8c2b7a6ca7d1ca780b3f83803cdc2082 --data-root DL3DV/1K/ -m output_DL3DV/1K/cd6c19a00fe93297c128769f81f7778f8c2b7a6ca7d1ca780b3f83803cdc2082/ --output-dir output_3d_bounding_dl3dv_evaluation_new --vllm-api-url http://100.102.160.232:8000/v1/chat/completions --mask-root sam_masks_debug_dl3dv/


# ========== josn2rerun_gyn.py 命令 ==========

# 场景1
python josn2rerun_gyn.py output_3d_bounding_dl3dv_evaluation_new/0bfab69b5b2b692d066b4435f2f6406357772aa53a3e17c3e0b8598083848a7a.json --data-root DL3DV/1K/   -m output_DL3DV/1K/0bfab69b5b2b692d066b4435f2f6406357772aa53a3e17c3e0b8598083848a7a/ --output-dir output_3d_bounding_dl3dv_evaluation_new  --no-include-desc

# 场景2
python josn2rerun_gyn.py output_3d_bounding_dl3dv_evaluation_new/2d52e65f703cf307013273ffeea79b76db153d665294becb16f00477aac05a88.json --data-root DL3DV/1K/   -m output_DL3DV/1K/2d52e65f703cf307013273ffeea79b76db153d665294becb16f00477aac05a88/ --output-dir output_3d_bounding_dl3dv_evaluation_new  --no-include-desc

# 场景3
python josn2rerun_gyn.py output_3d_bounding_dl3dv_evaluation_new/5b1e56386e263f2324365d2c623d6b746280aa4f5a6257bb0ba66e5dff135c8a.json --data-root DL3DV/1K/   -m output_DL3DV/1K/5b1e56386e263f2324365d2c623d6b746280aa4f5a6257bb0ba66e5dff135c8a/ --output-dir output_3d_bounding_dl3dv_evaluation_new  --no-include-desc

# 场景4
python josn2rerun_gyn.py output_3d_bounding_dl3dv_evaluation_new/7ae35f524e63aa86d1985f1c59dfd7a29cbd6069770ee26506ac26705917d9ea.json --data-root DL3DV/1K/   -m output_DL3DV/1K/7ae35f524e63aa86d1985f1c59dfd7a29cbd6069770ee26506ac26705917d9ea/ --output-dir output_3d_bounding_dl3dv_evaluation_new  --no-include-desc

# 场景5
python josn2rerun_gyn.py output_3d_bounding_dl3dv_evaluation_new/7ed574ca0081fa9a8c879a4b360ec7b274b3a9d248c1a6e66bd1e1d1c47bea4b.json --data-root DL3DV/1K/   -m output_DL3DV/1K/7ed574ca0081fa9a8c879a4b360ec7b274b3a9d248c1a6e66bd1e1d1c47bea4b/ --output-dir output_3d_bounding_dl3dv_evaluation_new  --no-include-desc

# 场景6
python josn2rerun_gyn.py output_3d_bounding_dl3dv_evaluation_new/7f5f6eac88ce535147bb9a7740635901929af1b08fbf0dd1adf28bf8aeee380a.json --data-root DL3DV/1K/   -m output_DL3DV/1K/7f5f6eac88ce535147bb9a7740635901929af1b08fbf0dd1adf28bf8aeee380a/ --output-dir output_3d_bounding_dl3dv_evaluation_new  --no-include-desc

# 场景7
python josn2rerun_gyn.py output_3d_bounding_dl3dv_evaluation_new/b3d3847b167ee770b105de4388108a1b8f3bfaa430d4d6ff0ba910d271d8a487.json --data-root DL3DV/1K/   -m output_DL3DV/1K/b3d3847b167ee770b105de4388108a1b8f3bfaa430d4d6ff0ba910d271d8a487/ --output-dir output_3d_bounding_dl3dv_evaluation_new  --no-include-desc

# 场景8
python josn2rerun_gyn.py output_3d_bounding_dl3dv_evaluation_new/c07ac2cdc0dddc8fe07c42258725ca0a5b486dcaccaef4fcbe8f5311cdee46fc.json --data-root DL3DV/1K/   -m output_DL3DV/1K/c07ac2cdc0dddc8fe07c42258725ca0a5b486dcaccaef4fcbe8f5311cdee46fc/ --output-dir output_3d_bounding_dl3dv_evaluation_new  --no-include-desc

# 场景9
python josn2rerun_gyn.py output_3d_bounding_dl3dv_evaluation_new/cd6c19a00fe93297c128769f81f7778f8c2b7a6ca7d1ca780b3f83803cdc2082.json --data-root DL3DV/1K/   -m output_DL3DV/1K/cd6c19a00fe93297c128769f81f7778f8c2b7a6ca7d1ca780b3f83803cdc2082/ --output-dir output_3d_bounding_dl3dv_evaluation_new  --no-include-desc


# ========== 3d_bounding_instance_gs_rerun_da3.py 命令 ==========

# 场景1
python 3d_bounding_instance_gs_rerun_da3.py --scene 0bfab69b5b2b692d066b4435f2f6406357772aa53a3e17c3e0b8598083848a7a --data-root DL3DV/1K/ -m output_DL3DV/1K/0bfab69b5b2b692d066b4435f2f6406357772aa53a3e17c3e0b8598083848a7a/ --output-dir output_3d_bounding_dl3dv_evaluation_da3  --vllm-api-url http://100.102.160.232:8000/v1/chat/completions --mask-root sam_masks_debug_dl3dv/

# 场景2
python 3d_bounding_instance_gs_rerun_da3.py --scene 2d52e65f703cf307013273ffeea79b76db153d665294becb16f00477aac05a88 --data-root DL3DV/1K/ -m output_DL3DV/1K/2d52e65f703cf307013273ffeea79b76db153d665294becb16f00477aac05a88/ --output-dir output_3d_bounding_dl3dv_evaluation_da3  --vllm-api-url http://100.102.160.232:8000/v1/chat/completions --mask-root sam_masks_debug_dl3dv/

# 场景3
python 3d_bounding_instance_gs_rerun_da3.py --scene 5b1e56386e263f2324365d2c623d6b746280aa4f5a6257bb0ba66e5dff135c8a --data-root DL3DV/1K/ -m output_DL3DV/1K/5b1e56386e263f2324365d2c623d6b746280aa4f5a6257bb0ba66e5dff135c8a/ --output-dir output_3d_bounding_dl3dv_evaluation_da3  --vllm-api-url http://100.102.160.232:8000/v1/chat/completions --mask-root sam_masks_debug_dl3dv/

# 场景4
python 3d_bounding_instance_gs_rerun_da3.py --scene 7ae35f524e63aa86d1985f1c59dfd7a29cbd6069770ee26506ac26705917d9ea --data-root DL3DV/1K/ -m output_DL3DV/1K/7ae35f524e63aa86d1985f1c59dfd7a29cbd6069770ee26506ac26705917d9ea/ --output-dir output_3d_bounding_dl3dv_evaluation_da3  --vllm-api-url http://100.102.160.232:8000/v1/chat/completions --mask-root sam_masks_debug_dl3dv/

# 场景5
python 3d_bounding_instance_gs_rerun_da3.py --scene 7ed574ca0081fa9a8c879a4b360ec7b274b3a9d248c1a6e66bd1e1d1c47bea4b --data-root DL3DV/1K/ -m output_DL3DV/1K/7ed574ca0081fa9a8c879a4b360ec7b274b3a9d248c1a6e66bd1e1d1c47bea4b/ --output-dir output_3d_bounding_dl3dv_evaluation_da3  --vllm-api-url http://100.102.160.232:8000/v1/chat/completions --mask-root sam_masks_debug_dl3dv/

# 场景6
python 3d_bounding_instance_gs_rerun_da3.py --scene 7f5f6eac88ce535147bb9a7740635901929af1b08fbf0dd1adf28bf8aeee380a --data-root DL3DV/1K/ -m output_DL3DV/1K/7f5f6eac88ce535147bb9a7740635901929af1b08fbf0dd1adf28bf8aeee380a/ --output-dir output_3d_bounding_dl3dv_evaluation_da3  --vllm-api-url http://100.102.160.232:8000/v1/chat/completions --mask-root sam_masks_debug_dl3dv/

# 场景7
python 3d_bounding_instance_gs_rerun_da3.py --scene b3d3847b167ee770b105de4388108a1b8f3bfaa430d4d6ff0ba910d271d8a487 --data-root DL3DV/1K/ -m output_DL3DV/1K/b3d3847b167ee770b105de4388108a1b8f3bfaa430d4d6ff0ba910d271d8a487/ --output-dir output_3d_bounding_dl3dv_evaluation_da3  --vllm-api-url http://100.102.160.232:8000/v1/chat/completions --mask-root sam_masks_debug_dl3dv/

# 场景8
python 3d_bounding_instance_gs_rerun_da3.py --scene c07ac2cdc0dddc8fe07c42258725ca0a5b486dcaccaef4fcbe8f5311cdee46fc --data-root DL3DV/1K/ -m output_DL3DV/1K/c07ac2cdc0dddc8fe07c42258725ca0a5b486dcaccaef4fcbe8f5311cdee46fc/ --output-dir output_3d_bounding_dl3dv_evaluation_da3  --vllm-api-url http://100.102.160.232:8000/v1/chat/completions --mask-root sam_masks_debug_dl3dv/

# 场景9
python 3d_bounding_instance_gs_rerun_da3.py --scene cd6c19a00fe93297c128769f81f7778f8c2b7a6ca7d1ca780b3f83803cdc2082 --data-root DL3DV/1K/ -m output_DL3DV/1K/cd6c19a00fe93297c128769f81f7778f8c2b7a6ca7d1ca780b3f83803cdc2082/ --output-dir output_3d_bounding_dl3dv_evaluation_da3  --vllm-api-url http://100.102.160.232:8000/v1/chat/completions --mask-root sam_masks_debug_dl3dv/