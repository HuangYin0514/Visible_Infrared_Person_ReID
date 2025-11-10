###########################################################
# Wandb
###########################################################
wandb login c74133df8c2cf575304acf8a99fe03ab74b6fe6a


############################################################
# Runs
############################################################
# python pre_process_sysu.py --data_path /kaggle/.../SYSU_MM01_concise111

# python main.py --config_file "config/method.yml" TASK.NOTES=V275 TASK.NAME=B_I_C_P OPTIMIZER.TOTAL_TRAIN_EPOCH=61 


python main.py --config_file "config/method.yml" TASK.NOTES=V294 TASK.NAME=B_I_C_P_regdb_8 OPTIMIZER.TOTAL_TRAIN_EPOCH=61 DATASET.TRAIN_DATASET=reg_db DATASET.TRAIN_DATASET_PATH=/kaggle/input/reg-db/RegDB/ DATASET.TRIAL=8 MODEL.NON_LOCAL_FLAG=False MODEL.MODAL_CALIBRATION_WEIGHT=0 MODEL.MODAL_PROPAGATION_WEIGHT=0
