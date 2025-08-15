###########################################################
# Wandb
###########################################################
wandb login c74133df8c2cf575304acf8a99fe03ab74b6fe6a


############################################################
# Runs
############################################################
# python pre_process_sysu.py --data_path /kaggle/.../SYSU_MM01_concise111

# python main.py --config_file "config/method.yml" TASK.NAME=Baseline TASK.NOTES=V007  MODEL.MODULE=Lucky

python main.py --config_file "config/method.yml" TASK.NAME=Baseline TASK.NOTES=V010  MODEL.MODULE=Lucky MODEL.BACKBONE_TYPE=resnet50

