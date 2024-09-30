# 第一批命令
#seed, device, kan_mlp, model
#kan pretrain, 90 e
bash ./bash_script/cl_scaling/finetune_cld64.sh 0 0 kan ./to_test/scale/rgcl_seed0_90_gnn1.pth > ./log/kan_pretrian.log 2>&1 &
#kan pretrain, 40 e
bash ./bash_script/cl_scaling/finetune_cld64.sh 0 1 kan ./to_test/scale/rgcl_seed0_40_gnn1.pth > ./log/kan_pretrian.log 2>&1 &
#kan w/o pretrain
bash ./bash_script/cl_scaling/finetune_cld64.sh 0 2 kan "" > ./log/kan_nopretrian.log 2>&1 &
#mlp w/o pretrain
bash ./bash_script/cl_scaling/finetune_cld64.sh 0 3 mlp "" > ./log/mlp_nopretrain.log 2>&1 &

#kan pretrain, 90 e
bash ./bash_script/cl_scaling/finetune_cld64.sh 1 4 kan ./to_test/scale/rgcl_seed0_90_gnn1.pth > ./log/kan_pretrian.log 2>&1 &
#kan pretrain, 40 e
bash ./bash_script/cl_scaling/finetune_cld64.sh 1 5 kan ./to_test/scale/rgcl_seed0_40_gnn1.pth > ./log/kan_pretrian.log 2>&1 &
#kan w/o pretrain
bash ./bash_script/cl_scaling/finetune_cld64.sh 1 6 kan "" > ./log/kan_nopretrian.log 2>&1 &
#mlp w/o pretrain
bash ./bash_script/cl_scaling/finetune_cld64.sh 1 7 mlp "" > ./log/mlp_nopretrain.log 2>&1 &

#kan pretrain, 90 e
bash ./bash_script/cl_scaling/finetune_cld64.sh 2 0 kan ./to_test/scale/rgcl_seed0_90_gnn1.pth > ./log/kan_pretrian.log 2>&1 &
#kan pretrain, 40 e
bash ./bash_script/cl_scaling/finetune_cld64.sh 2 1 kan ./to_test/scale/rgcl_seed0_40_gnn1.pth > ./log/kan_pretrian.log 2>&1 &
#kan w/o pretrain
bash ./bash_script/cl_scaling/finetune_cld64.sh 2 2 kan "" > ./log/kan_nopretrian.log 2>&1 &
#mlp w/o pretrain
bash ./bash_script/cl_scaling/finetune_cld64.sh 2 3 mlp "" > ./log/mlp_nopretrain.log 2>&1 &

#kan pretrain, 90 e
bash ./bash_script/cl_scaling/finetune_cld64.sh 3 4 kan ./to_test/scale/rgcl_seed0_90_gnn1.pth > ./log/kan_pretrian.log 2>&1 &
#kan pretrain, 40 e
bash ./bash_script/cl_scaling/finetune_cld64.sh 3 5 kan ./to_test/scale/rgcl_seed0_40_gnn1.pth > ./log/kan_pretrian.log 2>&1 &
#kan w/o pretrain
bash ./bash_script/cl_scaling/finetune_cld64.sh 3 6 kan "" > ./log/kan_nopretrian.log 2>&1 &
#mlp w/o pretrain
bash ./bash_script/cl_scaling/finetune_cld64.sh 3 7 mlp "" > ./log/mlp_nopretrain.log 2>&1 &