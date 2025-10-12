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

# python main.py --config_file "config/method.yml" TASK.NOTES=V217 TASK.NAME=B_I_C_P OPTIMIZER.TOTAL_TRAIN_EPOCH=61 

# python main.py --config_file "config/method.yml" TASK.NOTES=V219 TASK.NAME=B_I_C_P_regdb_1 OPTIMIZER.TOTAL_TRAIN_EPOCH=61 DATASET.TRAIN_DATASET=reg_db DATASET.TRAIN_DATASET_PATH=/kaggle/input/reg-db/RegDB/ DATASET.TRIAL=1
# python main.py --config_file "config/method.yml" TASK.NOTES=V220 TASK.NAME=B_I_C_P_regdb_2 OPTIMIZER.TOTAL_TRAIN_EPOCH=61 DATASET.TRAIN_DATASET=reg_db DATASET.TRAIN_DATASET_PATH=/kaggle/input/reg-db/RegDB/ DATASET.TRIAL=2
# python main.py --config_file "config/method.yml" TASK.NOTES=V220 TASK.NAME=B_I_C_P_regdb_3 OPTIMIZER.TOTAL_TRAIN_EPOCH=61 DATASET.TRAIN_DATASET=reg_db DATASET.TRAIN_DATASET_PATH=/kaggle/input/reg-db/RegDB/ DATASET.TRIAL=3
# python main.py --config_file "config/method.yml" TASK.NOTES=V220 TASK.NAME=B_I_C_P_regdb_4 OPTIMIZER.TOTAL_TRAIN_EPOCH=61 DATASET.TRAIN_DATASET=reg_db DATASET.TRAIN_DATASET_PATH=/kaggle/input/reg-db/RegDB/ DATASET.TRIAL=4
# python main.py --config_file "config/method.yml" TASK.NOTES=V220 TASK.NAME=B_I_C_P_regdb_5 OPTIMIZER.TOTAL_TRAIN_EPOCH=61 DATASET.TRAIN_DATASET=reg_db DATASET.TRAIN_DATASET_PATH=/kaggle/input/reg-db/RegDB/ DATASET.TRIAL=5
# python main.py --config_file "config/method.yml" TASK.NOTES=V220 TASK.NAME=B_I_C_P_regdb_6 OPTIMIZER.TOTAL_TRAIN_EPOCH=61 DATASET.TRAIN_DATASET=reg_db DATASET.TRAIN_DATASET_PATH=/kaggle/input/reg-db/RegDB/ DATASET.TRIAL=6
# python main.py --config_file "config/method.yml" TASK.NOTES=V220 TASK.NAME=B_I_C_P_regdb_7 OPTIMIZER.TOTAL_TRAIN_EPOCH=61 DATASET.TRAIN_DATASET=reg_db DATASET.TRAIN_DATASET_PATH=/kaggle/input/reg-db/RegDB/ DATASET.TRIAL=7
# python main.py --config_file "config/method.yml" TASK.NOTES=V220 TASK.NAME=B_I_C_P_regdb_8 OPTIMIZER.TOTAL_TRAIN_EPOCH=61 DATASET.TRAIN_DATASET=reg_db DATASET.TRAIN_DATASET_PATH=/kaggle/input/reg-db/RegDB/ DATASET.TRIAL=8
# python main.py --config_file "config/method.yml" TASK.NOTES=V220 TASK.NAME=B_I_C_P_regdb_9 OPTIMIZER.TOTAL_TRAIN_EPOCH=61 DATASET.TRAIN_DATASET=reg_db DATASET.TRAIN_DATASET_PATH=/kaggle/input/reg-db/RegDB/ DATASET.TRIAL=9
# python main.py --config_file "config/method.yml" TASK.NOTES=V220 TASK.NAME=B_I_C_P_regdb_10 OPTIMIZER.TOTAL_TRAIN_EPOCH=61 DATASET.TRAIN_DATASET=reg_db DATASET.TRAIN_DATASET_PATH=/kaggle/input/reg-db/RegDB/ DATASET.TRIAL=10


python main.py --config_file "config/method.yml" TASK.NOTES=V224 TASK.NAME=B_I_C_P_regdb_8 OPTIMIZER.TOTAL_TRAIN_EPOCH=120 DATASET.TRAIN_DATASET=reg_db DATASET.TRAIN_DATASET_PATH=/kaggle/input/reg-db/RegDB/ DATASET.TRIAL=8 MODEL.MODAL_INTERACTION_FLAG=False MODEL.MODAL_CALIBRATION_FLAG=False MODEL.MODAL_PROPAGATION_WEIGHT=0.001
