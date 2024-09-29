# 第一批命令
#seed, device, kan_mlp, model
#kan pretrain, 90 e
bash ./bash_script/cl_scaling/finetune_cld64.sh 1 2 kan ./to_test/scale/rgcl_seed0_90_gnn1.pth > ./log/kan_pretrian.log 2>&1 &
#kan pretrain, 40 e
bash ./bash_script/cl_scaling/finetune_cld64.sh 1 3 kan ./to_test/scale/rgcl_seed0_40_gnn1.pth > ./log/kan_pretrian.log 2>&1 &
#kan w/o pretrain
bash ./bash_script/cl_scaling/finetune_cld64.sh 1 4 kan "" > ./log/kan_nopretrian.log 2>&1 &
#mlp w/o pretrain
bash ./bash_script/cl_scaling/finetune_cld64.sh 1 5 mlp "" > ./log/mlp_nopretrain.log 2>&1 &

#kan pretrain, 90 e
bash ./bash_script/cl_scaling/finetune_cld64.sh 2 2 kan ./to_test/scale/rgcl_seed0_90_gnn1.pth > ./log/kan_pretrian.log 2>&1 &
#kan pretrain, 40 e
bash ./bash_script/cl_scaling/finetune_cld64.sh 2 3 kan ./to_test/scale/rgcl_seed0_40_gnn1.pth > ./log/kan_pretrian.log 2>&1 &
#kan w/o pretrain
bash ./bash_script/cl_scaling/finetune_cld64.sh 2 4 kan "" > ./log/kan_nopretrian.log 2>&1 &
#mlp w/o pretrain
bash ./bash_script/cl_scaling/finetune_cld64.sh 2 5 mlp "" > ./log/mlp_nopretrain.log 2>&1 &
