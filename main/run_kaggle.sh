###########################################################
# Wandb
###########################################################
wandb login c74133df8c2cf575304acf8a99fe03ab74b6fe6a


############################################################
# Runs
############################################################
# python pre_process_sysu.py --data_path /kaggle/.../SYSU_MM01_concise111


python main.py --config_file "config/method.yml" TASK.NOTES=V103 TASK.NAME=Baseline OPTIMIZER.TOTAL_TRAIN_EPOCH=61 MODEL.MODAL_INTERACTION_FLAG=False MODEL.MODAL_CALIBRATION_FLAG=False MODEL.MODAL_PROPAGATION_FALG=False

#  OPTIMIZER.LEARNING_RATE=0.0003