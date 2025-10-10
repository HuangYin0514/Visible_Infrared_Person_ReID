###########################################################
# Wandb
###########################################################
wandb login c74133df8c2cf575304acf8a99fe03ab74b6fe6a


############################################################
# Runs
############################################################
# python pre_process_sysu.py --data_path /kaggle/.../SYSU_MM01_concise111
#  OPTIMIZER.LEARNING_RATE=0.0003

# python main.py --config_file "config/method.yml" TASK.NOTES=V106 TASK.NAME=B_IP OPTIMIZER.TOTAL_TRAIN_EPOCH=61 MODEL.MODAL_INTERACTION_FLAG=False MODEL.MODAL_CALIBRATION_FLAG=False MODEL.MODAL_PROPAGATION_FALG=True

# python main.py --config_file "config/method.yml" TASK.NOTES=V212 TASK.NAME=B_I_C_P OPTIMIZER.TOTAL_TRAIN_EPOCH=61 

python main.py --config_file "config/method.yml" TASK.NOTES=V214 TASK.NAME=B_I_C_P_regdb OPTIMIZER.TOTAL_TRAIN_EPOCH=61 DATASET.TRAIN_DATASET=reg_db DATASET.TRAIN_DATASET_PATH=/kaggle/input/reg-db/RegDB/ DATASET.TRIAL=1
