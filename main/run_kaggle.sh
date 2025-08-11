###########################################################
# Wandb
###########################################################
wandb login c74133df8c2cf575304acf8a99fe03ab74b6fe6a

###########################################################
# Runs
###########################################################
# python main.py --config_file "config/method.yml" TASK.NOTES=v859_visualization TASK.NAME=R_H2GP_S2GA_M2PQF_C2Mkl

# python main.py --config_file "config/method.yml" TASK.NOTES=v859_visualization TASK.NAME=R_H2GP_S2GA_M2PQF_C2Mkl MODEL.MODULE=Baseline

###########################################################
# Visualize
###########################################################
mkdir -p results/outputs/models
wget -O results/outputs/models/model_114.pth "https://storage.googleapis.com/kaggle-script-versions/250844267/output/reid_multiview/main/results/outputs/models/model_114.pth?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20250728%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250728T150718Z&X-Goog-Expires=3600&X-Goog-SignedHeaders=host&X-Goog-Signature=bed56c5d5ba7d3b67094d66a74e1986732fb33a25b197c778447d220965f0ff296b151540c96c04fc9b610ae501651153585db5daa18e589e005838f5254341e34412884099ea02b2f055181a6e524de0bad303c6fe9d964f15c00001c2043836180afe80e7572728e2375d4ce199660192a2ff48a17988406a4bdb2d09c8ce53528ec322f4ec77d92f09d12c857b1fee2a84118a7cb041f838bfbe62e0b49af38b6f8de38694c279db96a8f0b055fe3962456f98b02d2e326dedae0a378f14fcd13658668b64eb650fe760da0aa0c89fd74b1812d1b16d6e6d1a9997011ae3e3c1c9db39f796a4840482bc9e6cd2ac1c4b9e2a97f53e241e942f5645a2bb333"

# python main.py --config_file "config/method.yml" TASK.NAME=R_H2GP_S2GA_M2PQF_C2Mkl TASK.NOTES=v862_Baseline_visualization TASK.MODE=visualization MODEL.MODULE=Baseline TEST.RESUME_TEST_MODEL=114 
python main.py --config_file "config/method.yml" TASK.NAME=R_H2GP_S2GA_M2PQF_C2Mkl TASK.NOTES=v863_Lucky_visualization TASK.MODE=visualization MODEL.MODULE=Lucky TEST.RESUME_TEST_MODEL=114 

tar -czf results.tar.gz results
rm -rf results/outputs/actmap results/outputs/rank results/outputs/tSNE 
