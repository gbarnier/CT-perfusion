################################################################################
################################# Training subset 5 ############################
################################################################################
# Training/dev/test on one full slice
train6_file=par/train_file6.txt
dev6_file=par/dev_file6.txt
test6_file=par/test_file6.txt

################################## Baseline ####################################
train6-base1:
	rm -rf models/train6-base1
	python ./python/CTP_main.py train train6-base1 --model baseline --n_epochs 100 --lr 1e-6 --device cuda:0 --train_file_list ${train6_file} --dev_file_list ${dev6_file} --baseline_n_hidden 100

#################################### FC ########################################
train6-debug-1:
	rm -rf models/train6-debug
	python ./python/CTP_main.py train train6-debug --model fc6 --n_epochs 5 --lr 0.001 --device cuda:1 --train_file_list ${train6_file} --dev_file_list ${dev6_file} --batch_size 32 --seed 10

train6-debug-2:
	rm -rf models/train6-debug2
	python ./python/CTP_main.py train train6-debug2 --model fc6 --n_epochs 20 --lr 0.001 --device cuda:1 --train_file_list ${train6_file} --dev_file_list ${dev6_file} --batch_size 2048 --seed 10

train6-debug-3:
	rm -rf models/train6-debug3
	python ./python/CTP_main.py train train6-debug3 --model fc6 --n_epochs 20 --lr 0.001 --device cuda:1 --train_file_list ${train6_file} --dev_file_list ${dev6_file} --batch_size 2048 --seed 10

train6-fc1:
	rm -rf models/train6-fc1
	python ./python/CTP_main.py train train6-fc1 --model fc6 --n_epochs 20 --lr 0.001 --device cuda:1 --train_file_list ${train6_file} --dev_file_list ${dev6_file} --batch_size 512

train6-fc2:
	rm -rf models/train6-fc2
	python ./python/CTP_main.py train train6-fc2 --model fc6 --n_epochs 20 --lr 1.0e-7 --device cuda:1 --train_file_list ${train6_file} --dev_file_list ${dev6_file} --batch_size 512

train6-fc3:
	rm -rf models/train6-fc3
	python ./python/CTP_main.py train train6-fc3 --model fc6 --n_epochs 20 --lr 1.0e-6 --device cuda:2 --train_file_list ${train6_file} --dev_file_list ${dev6_file} --batch_size 512

train6-fc4:
	rm -rf models/train6-fc4
	python ./python/CTP_main.py train train6-fc4 --model fc6 --n_epochs 20 --lr 1.0e-5 --device cuda:3 --train_file_list ${train6_file} --dev_file_list ${dev6_file} --batch_size 512

train6-fc5:
	rm -rf models/train6-fc5
	python ./python/CTP_main.py train train6-fc5 --model fc6 --n_epochs 20 --lr 1.0e-4 --device cuda:0 --train_file_list ${train6_file} --dev_file_list ${dev6_file} --batch_size 512

train6-fc6:
	rm -rf models/train6-fc6
	python ./python/CTP_main.py train train6-fc6 --model fc6 --n_epochs 20 --lr 1.0e-3 --device cuda:1 --train_file_list ${train6_file} --dev_file_list ${dev6_file} --batch_size 512

train6-fc7:
	rm -rf models/train6-fc7
	python ./python/CTP_main.py train train6-fc7 --model fc6 --n_epochs 20 --lr 1.0e-2 --device cuda:2 --train_file_list ${train6_file} --dev_file_list ${dev6_file} --batch_size 512

train6-fc8:
	rm -rf models/train6-fc8
	python ./python/CTP_main.py train train6-fc8 --model fc6 --n_epochs 20 --lr 1.5e-2 --device cuda:2 --train_file_list ${train6_file} --dev_file_list ${dev6_file} --batch_size 512

#################################### Kir #######################################
train6-kir1:
	rm -rf models/train6-kir1
	python ./python/CTP_main.py train train6-kir1 --model kiranyaz --n_epochs 20 --lr 1.e-3 --device cuda:2 --train_file_list ${train6_file} --dev_file_list ${dev6_file} --batch_size 512

train6-kir2:
	rm -rf models/train6-kir2
	python ./python/CTP_main.py train train6-kir2 --model kiranyaz --n_epochs 20 --lr 1.0e-6 --device cuda:0 --train_file_list ${train6_file} --dev_file_list ${dev6_file} --batch_size 512

train6-kir3:
	rm -rf models/train6-kir3
	python ./python/CTP_main.py train train6-kir3 --model kiranyaz --n_epochs 20 --lr 1.0e-5 --device cuda:1 --train_file_list ${train6_file} --dev_file_list ${dev6_file} --batch_size 512

train6-kir4:
	rm -rf models/train6-kir4
	python ./python/CTP_main.py train train6-kir4 --model kiranyaz --n_epochs 20 --lr 1.0e-4 --device cuda:2 --train_file_list ${train6_file} --dev_file_list ${dev6_file} --batch_size 512

train6-kir5:
	rm -rf models/train6-kir5
	python ./python/CTP_main.py train train6-kir5 --model kiranyaz --n_epochs 20 --lr 1.0e-2 --device cuda:3 --train_file_list ${train6_file} --dev_file_list ${dev6_file} --batch_size 512

train6-kir6:
	rm -rf models/train6-kir6
	python ./python/CTP_main.py train train6-kir6 --model kiranyaz --n_epochs 20 --lr 1.0e-1 --device cuda:0 --train_file_list ${train6_file} --dev_file_list ${dev6_file} --batch_size 512

train6-kir7:
	rm -rf models/train6-kir7
	python ./python/CTP_main.py train train6-kir7 --model kiranyaz --n_epochs 20 --lr 1.0e-6 --device cuda:0 --train_file_list ${train6_file} --dev_file_list ${dev6_file} --batch_size 512 --lr_decay step --step_size 5 --decay_gamma 0.8

#################################### Kir #######################################
train6-gui1:
	rm -rf models/train6-gui1
	python ./python/CTP_main.py train train6-gui1 --model gui --n_epochs 20 --lr 1.0e-6 --device cuda:1 --train_file_list ${train6_file} --dev_file_list ${dev6_file} --batch_size 512

train6-gui2:
	rm -rf models/train6-gui2
	python ./python/CTP_main.py train train6-gui2 --model gui --n_epochs 20 --lr 1.0e-5 --device cuda:1 --train_file_list ${train6_file} --dev_file_list ${dev6_file} --batch_size 512

train6-gui3:
	rm -rf models/train6-gui3
	python ./python/CTP_main.py train train6-gui3 --model gui --n_epochs 20 --lr 1.0e-4 --device cuda:3 --train_file_list ${train6_file} --dev_file_list ${dev6_file} --batch_size 512

train6-gui4:
	rm -rf models/train6-gui4
	python ./python/CTP_main.py train train6-gui4 --model gui --n_epochs 20 --lr 1.0e-3 --device cuda:2 --train_file_list ${train6_file} --dev_file_list ${dev6_file} --batch_size 512

train6-gui5:
	rm -rf models/train6-gui5
	python ./python/CTP_main.py train train6-gui5 --model gui --n_epochs 20 --lr 1.0e-2 --device cuda:3 --train_file_list ${train6_file} --dev_file_list ${dev6_file} --batch_size 512

train6-gui6:
	rm -rf models/train6-gui6
	python ./python/CTP_main.py train train6-gui6 --model gui --n_epochs 20 --lr 1.0e-5 --device cuda:1 --train_file_list ${train6_file} --dev_file_list ${dev6_file} --batch_size 512 --lr_decay step --step_size 10 --decay_gamma 0.9

train6-gui7:
	rm -rf models/train6-gui7
	python ./python/CTP_main.py train train6-gui7 --model gui --n_epochs 20 --lr 1.0e-5 --device cuda:0 --train_file_list ${train6_file} --dev_file_list ${dev6_file} --batch_size 512 --lr_decay step --step_size 4 --decay_gamma 0.5

#################################### Blog med ##################################
train6-blogMed1:
	rm -rf models/train6-blogMed1
	python ./python/CTP_main.py train train6-blogMed1 --model blogMed --n_epochs 20 --lr=0.001 --device cuda:3 --train_file_list ${train6_file} --dev_file_list ${dev6_file} --batch_size 512

#################################### Blog deep #################################
train6-blogDeep1:
	rm -rf models/train6-blogDeep1
	python ./python/CTP_main.py train train6-blogDeep1 --model blogDeep --n_epochs 50 --lr=1.0e-7 --device cuda:0 --train_file_list ${train6_file} --dev_file_list ${dev6_file} --batch_size 1024

train6-blogDeep2:
	rm -rf models/train6-blogDeep2
	python ./python/CTP_main.py train train6-blogDeep2 --model blogDeep --n_epochs 50 --lr=1.0e-6 --device cuda:0 --train_file_list ${train6_file} --dev_file_list ${dev6_file} --batch_size 1024

train6-blogDeep3:
	rm -rf models/train6-blogDeep3
	python ./python/CTP_main.py train train6-blogDeep3 --model blogDeep --n_epochs 50 --lr=1.0e-5 --device cuda:1 --train_file_list ${train6_file} --dev_file_list ${dev6_file} --batch_size 1024

train6-blogDeep4:
	rm -rf models/train6-blogDeep4
	python ./python/CTP_main.py train train6-blogDeep4 --model blogDeep --n_epochs 50 --lr=1.0e-4 --device cuda:1 --train_file_list ${train6_file} --dev_file_list ${dev6_file} --batch_size 1024

train6-blogDeep5:
	rm -rf models/train6-blogDeep5
	python ./python/CTP_main.py train train6-blogDeep5 --model blogDeep --n_epochs 50 --lr=1.0e-3 --device cuda:2 --train_file_list ${train6_file} --dev_file_list ${dev6_file} --batch_size 1024

train6-blogDeep6:
	rm -rf models/train6-blogDeep6
	python ./python/CTP_main.py train train6-blogDeep6 --model blogDeep --n_epochs 50 --lr=1.0e-2 --device cuda:2 --train_file_list ${train6_file} --dev_file_list ${dev6_file} --batch_size 1024

train6-blogDeep7:
	rm -rf models/train6-blogDeep7
	python ./python/CTP_main.py train train6-blogDeep7 --model blogDeep --n_epochs 50 --lr=1.0e-1 --device cuda:2 --train_file_list ${train6_file} --dev_file_list ${dev6_file} --batch_size 1024

train6-blogDeep8:
	rm -rf models/train6-blogDeep8
	python ./python/CTP_main.py train train6-blogDeep8 --model blogDeep --n_epochs 50 --lr=1.0e-7 --device cuda:3 --train_file_list ${train6_file} --dev_file_list ${dev6_file} --batch_size 1024 --lr_decay decay

train6-blogDeep9:
	rm -rf models/train6-blogDeep9
	python ./python/CTP_main.py train train6-blogDeep9 --model blogDeep --n_epochs 50 --lr=1.0e-6 --device cuda:3 --train_file_list ${train6_file} --dev_file_list ${dev6_file} --batch_size 1024 --lr_decay decay

train6-blogDeep10:
	rm -rf models/train6-blogDeep10
	python ./python/CTP_main.py train train6-blogDeep10 --model blogDeep --n_epochs 200 --lr=1.0e-6 --device cuda:0 --train_file_list ${train6_file} --dev_file_list ${dev6_file} --batch_size 1024 --lr_decay decay --decay_rate 0.01

train6-blogDeep11:
	rm -rf models/train6-blogDeep11
	python ./python/CTP_main.py train train6-blogDeep11 --model blogDeep --n_epochs 200 --lr=1.0e-6 --device cuda:0 --train_file_list ${train6_file} --dev_file_list ${dev6_file} --batch_size 1024 --lr_decay decay --decay_rate 0.005

train6-blogDeep12:
	rm -rf models/train6-blogDeep12
	python ./python/CTP_main.py train train6-blogDeep12 --model blogDeep --n_epochs 200 --lr=1.0e-6 --device cuda:1 --train_file_list ${train6_file} --dev_file_list ${dev6_file} --batch_size 1024 --lr_decay exp --decay_gamma 0.99

train6-blogDeep13:
	rm -rf models/train6-blogDeep13
	python ./python/CTP_main.py train train6-blogDeep13 --model blogDeep --n_epochs 200 --lr=1.0e-6 --device cuda:1 --train_file_list ${train6_file} --dev_file_list ${dev6_file} --batch_size 1024 --lr_decay exp --decay_gamma 0.995

train6-blogDeep14:
	rm -rf models/train6-blogDeep14
	python ./python/CTP_main.py train train6-blogDeep14 --model blogDeep --n_epochs 200 --lr=1.0e-6 --device cuda:3 --train_file_list ${train6_file} --dev_file_list ${dev6_file} --batch_size 1024 --lr_decay exp --decay_gamma 0.999

train6-blogDeep15:
	rm -rf models/train6-blogDeep15
	python ./python/CTP_main.py train train6-blogDeep15 --model blogDeep --n_epochs 200 --lr=1.0e-6 --device cuda:3 --train_file_list ${train6_file} --dev_file_list ${dev6_file} --batch_size 1024 --lr_decay step --step_size 10 --decay_gamma 0.99

train6-blogDeep16:
	rm -rf models/train6-blogDeep16
	python ./python/CTP_main.py train train6-blogDeep16 --model blogDeep --n_epochs 200 --lr=1.0e-6 --device cuda:3 --train_file_list ${train6_file} --dev_file_list ${dev6_file} --batch_size 1024 --lr_decay step --step_size 10 --decay_gamma 0.995

train6-blogDeep17:
	rm -rf models/train6-blogDeep17
	python ./python/CTP_main.py train train6-blogDeep17 --model blogDeep --n_epochs 200 --lr=1.0e-6 --device cuda:2 --train_file_list ${train6_file} --dev_file_list ${dev6_file} --batch_size 1024 --lr_decay step --step_size 10 --decay_gamma 0.999

train6-blogDeep18:
	rm -rf models/train6-blogDeep18
	python ./python/CTP_main.py train train6-blogDeep18 --model blogDeep --n_epochs 200 --lr=1.0e-6 --device cuda:2 --train_file_list ${train6_file} --dev_file_list ${dev6_file} --batch_size 1024 --lr_decay step --step_size 10 --decay_gamma 0.999

#################################### Blog small ##################################
train6-blogSmall1:
	rm -rf models/train6-blogSmall1
	python ./python/CTP_main.py train train6-blogSmall1 --model blogSmall --n_epochs 20 --lr=0.001 --device cuda:0 --train_file_list ${train6_file} --dev_file_list ${dev6_file} --batch_size 1024

#################################### Blog Large ################################
train6-blogLarge1:
	rm -rf models/train6-blogLarge1
	python ./python/CTP_main.py train train6-blogLarge1 --model blogLarge --n_epochs 20 --lr=1.0e-7 --device cuda:0 --train_file_list ${train6_file} --dev_file_list ${dev6_file} --batch_size 1024

train6-blogLarge2:
	rm -rf models/train6-blogLarge2
	python ./python/CTP_main.py train train6-blogLarge2 --model blogLarge --n_epochs 20 --lr=1.0e-6 --device cuda:1 --train_file_list ${train6_file} --dev_file_list ${dev6_file} --batch_size 1024

train6-blogLarge3:
	rm -rf models/train6-blogLarge3
	python ./python/CTP_main.py train train6-blogLarge3 --model blogLarge --n_epochs 20 --lr=1.0e-5 --device cuda:2 --train_file_list ${train6_file} --dev_file_list ${dev6_file} --batch_size 1024

train6-blogLarge4:
	rm -rf models/train6-blogLarge4
	python ./python/CTP_main.py train train6-blogLarge4 --model blogLarge --n_epochs 20 --lr=1.0e-4 --device cuda:3 --train_file_list ${train6_file} --dev_file_list ${dev6_file} --batch_size 1024

train6-blogLarge5:
	rm -rf models/train6-blogLarge5
	python ./python/CTP_main.py train train6-blogLarge5 --model blogLarge --n_epochs 20 --lr=1.0e-3 --device cuda:0 --train_file_list ${train6_file} --dev_file_list ${dev6_file} --batch_size 1024

train6-blogLarge6:
	rm -rf models/train6-blogLarge6
	python ./python/CTP_main.py train train6-blogLarge6 --model blogLarge --n_epochs 20 --lr=1.0e-2 --device cuda:1 --train_file_list ${train6_file} --dev_file_list ${dev6_file} --batch_size 1024

train6-blogLarge7:
	rm -rf models/train6-blogLarge7
	python ./python/CTP_main.py train train6-blogLarge7 --model blogLarge --n_epochs 20 --lr=0.1 --device cuda:2 --train_file_list ${train6_file} --dev_file_list ${dev6_file} --batch_size 1024

train6-blogLarge8:
	rm -rf models/train6-blogLarge8
	python ./python/CTP_main.py train train6-blogLarge8 --model blogLarge --n_epochs 20 --lr=1.0 --device cuda:3 --train_file_list ${train6_file} --dev_file_list ${dev6_file} --batch_size 1024

train6-blogLarge9:
	rm -rf models/train6-blogLarge9
	python ./python/CTP_main.py train train6-blogLarge9 --model blogLarge --n_epochs 20 --lr=1.0e-7 --device cuda:0 --train_file_list ${train6_file} --dev_file_list ${dev6_file} --batch_size 1024 --lr_decay step --step_size 4 --decay_gamma 0.5

train6-blogLarge10:
	rm -rf models/train6-blogLarge10
	python ./python/CTP_main.py train train6-blogLarge10 --model blogLarge --n_epochs 20 --lr=1.0e-6 --device cuda:1 --train_file_list ${train6_file} --dev_file_list ${dev6_file} --batch_size 1024 --lr_decay step --step_size 4 --decay_gamma 0.5

train6-blogLarge11:
	rm -rf models/train6-blogLarge11
	python ./python/CTP_main.py train train6-blogLarge11 --model blogLarge --n_epochs 20 --lr=1.0e-5 --device cuda:2 --train_file_list ${train6_file} --dev_file_list ${dev6_file} --batch_size 1024 --lr_decay step --step_size 4 --decay_gamma 0.5

train6-blogLarge12:
	rm -rf models/train6-blogLarge12
	python ./python/CTP_main.py train train6-blogLarge12 --model blogLarge --n_epochs 20 --lr=1.0e-4 --device cuda:2 --train_file_list ${train6_file} --dev_file_list ${dev6_file} --batch_size 1024 --lr_decay step --step_size 4 --decay_gamma 0.5

train6-blogLarge13:
	rm -rf models/train6-blogLarge13
	python ./python/CTP_main.py train train6-blogLarge13 --model blogLarge --n_epochs 20 --lr=1.0e-4 --device cuda:2 --train_file_list ${train6_file} --dev_file_list ${dev6_file} --batch_size 1024 --lr_decay step --step_size 4 --decay_gamma 0.5

################################################################################
############################ Prediction on train data ##########################
################################################################################
pred6_file_train=dat/S00242.h5

pred6-base%:
	python ./python/CTP_main.py predict train6-base$* --model baseline --patient_file ${pred6_file_train} --patient_id S00243 --baseline_n_hidden 100 --device cuda:0

pred6-fc%:
	python ./python/CTP_main.py predict train6-fc$* --model fc6 --patient_file ${pred6_file_train} --patient_id S00243 --device cuda:0

pred6-kir%:
	python ./python/CTP_main.py predict train6-kir$* --model kiranyaz --patient_file ${pred6_file_train} --patient_id S00243 --device cuda:1

pred6-gui%:
	python ./python/CTP_main.py predict train6-gui$* --model gui --patient_file ${pred6_file_train} --patient_id S00243 --device cuda:1

pred6-blogMed%:
	python ./python/CTP_main.py predict train6-blogMed$* --model blogMed --patient_file ${pred6_file_train} --patient_id S00243 --device cuda:2

pred6-blogSmall%:
	python ./python/CTP_main.py predict train6-blogSmall$* --model blogSmall --patient_file ${pred6_file_train} --patient_id S00243 --device cuda:2

pred6-blogDeep%:
	python ./python/CTP_main.py predict train6-blogDeep$* --model blogDeep --patient_file ${pred6_file_train} --patient_id S00243 --device cuda:2

makePred6:
	# make pred6-base1 -B
	# make pred6-fc1 -B
	# make pred6-kir1 -B
	# make pred6-kir2 -B
	# make pred6-kir3 -B
	# make pred6-kir4 -B
	# make pred6-kir5 -B
	# make pred6-kir6 -B
	# make pred6-gui1 -B
	# make pred6-gui2 -B
	# make pred6-gui3 -B
	# make pred6-gui4 -B
	# make pred6-gui5 -B
	# make pred6-blogMed1 -B
	# make pred6-blogDeep1 -B
	# make pred6-blogDeep2 -B
	# make pred6-blogDeep3 -B
	# make pred6-blogDeep4 -B
	# make pred6-blogDeep5 -B
	# make pred6-blogDeep6 -B
	# make pred6-blogDeep7 -B
	# make pred6-blogDeep8 -B
	# make pred6-blogDeep9 -B
	# make pred6-blogDeep10 -B
	# make pred6-blogDeep11 -B
	# make pred6-blogDeep12 -B
	# make pred6-blogDeep13 -B
	make pred6-blogDeep14 -B
	make pred6-blogDeep15 -B
	make pred6-blogDeep16 -B
	# make pred6-blogDeep17 -B

################################################################################
################################# Display ######################################
################################################################################
eclip6=300
bclip6=0

dispResult6-s%:
	# Training data
	Window3d n3=1 f3=$* < ${pred6_file_train}_tmax_m.H > tTrain.H
	# Base
	# Window3d n3=1 f3=$* < models/train6-base1/train6-base1_S00243_pred_tmax_sep.H > tBase1.H
	# # Cat axis=3 tTrain.H tBase1.H | Grey color=j newclip=1 bclip=${bclip6} eclip=${eclip6} grid=y titles="Train:Base1" gainpanel=a | Xtpen pixmaps=y &
	# # Fc
	# Window3d n3=1 f3=$* < models/train6-fc1/train6-fc1_S00243_pred_tmax_sep.H > tFc1.H
	# # Cat axis=3 tTrain.H tFc1.H | Grey color=j newclip=1 bclip=${bclip6} eclip=${eclip6} grid=y titles="Train:Fc1" gainpanel=a | Xtpen pixmaps=y &
	# # Kir
	# Window3d n3=1 f3=$* < models/train6-kir1/train6-kir1_S00243_pred_tmax_sep.H > tKir1.H
	# Window3d n3=1 f3=$* < models/train6-kir2/train6-kir2_S00243_pred_tmax_sep.H > tKir2.H
	# Window3d n3=1 f3=$* < models/train6-kir3/train6-kir3_S00243_pred_tmax_sep.H > tKir3.H
	# Window3d n3=1 f3=$* < models/train6-kir4/train6-kir4_S00243_pred_tmax_sep.H > tKir4.H
	# Window3d n3=1 f3=$* < models/train6-kir5/train6-kir5_S00243_pred_tmax_sep.H > tKir5.H
	# Window3d n3=1 f3=$* < models/train6-kir6/train6-kir6_S00243_pred_tmax_sep.H > tKir6.H
	# Cat axis=3 tTrain.H tKir1.H tKir2.H tKir3.H tKir4.H tKir5.H tKir6.H | Grey color=j newclip=1 bclip=${bclip6} eclip=${eclip6} grid=y titles="Train:Kir1:Kir2:Kir3:Kir4:Kir5:Kir6" gainpanel=a | Xtpen pixmaps=y &
	# # Gui
	# Window3d n3=1 f3=$* < models/train6-gui1/train6-gui1_S00243_pred_tmax_sep.H > tGui1.H
	# Window3d n3=1 f3=$* < models/train6-gui2/train6-gui2_S00243_pred_tmax_sep.H > tGui2.H
	# Window3d n3=1 f3=$* < models/train6-gui3/train6-gui3_S00243_pred_tmax_sep.H > tGui3.H
	# Window3d n3=1 f3=$* < models/train6-gui4/train6-gui4_S00243_pred_tmax_sep.H > tGui4.H
	# Window3d n3=1 f3=$* < models/train6-gui5/train6-gui5_S00243_pred_tmax_sep.H > tGui5.H
	# Cat axis=3 tTrain.H tGui1.H tGui2.H tGui3.H tGui4.H tGui5.H | Grey color=j newclip=1 bclip=${bclip6} eclip=${eclip6} grid=y titles="Train:Gui1:Gui2:Gui3:Gui4:Gui5" gainpanel=a | Xtpen pixmaps=y &
	# BlogMed
	# Window3d n3=1 f3=$* < models/train6-blogMed1/train6-blogMed1_S00243_pred_tmax_sep.H > tBlogMed1.H
	# Cat axis=3 tTrain.H tBlogMed1.H | Grey color=j newclip=1 bclip=${bclip6} eclip=${eclip6} grid=y titles="Train:BlogMed1" gainpanel=a | Xtpen pixmaps=y &
	# Blog Deep
	# Window3d n3=1 f3=$* < models/train6-blogDeep1/train6-blogDeep1_S00243_pred_tmax_sep.H > tBlogDeep1.H
	# Window3d n3=1 f3=$* < models/train6-blogDeep2/train6-blogDeep2_S00243_pred_tmax_sep.H > tBlogDeep2.H
	# Window3d n3=1 f3=$* < models/train6-blogDeep3/train6-blogDeep3_S00243_pred_tmax_sep.H > tBlogDeep3.H
	# Window3d n3=1 f3=$* < models/train6-blogDeep4/train6-blogDeep4_S00243_pred_tmax_sep.H > tBlogDeep4.H
	# Window3d n3=1 f3=$* < models/train6-blogDeep5/train6-blogDeep5_S00243_pred_tmax_sep.H > tBlogDeep5.H
	# Window3d n3=1 f3=$* < models/train6-blogDeep6/train6-blogDeep6_S00243_pred_tmax_sep.H > tBlogDeep6.H
	# Window3d n3=1 f3=$* < models/train6-blogDeep7/train6-blogDeep7_S00243_pred_tmax_sep.H > tBlogDeep7.H
	# Window3d n3=1 f3=$* < models/train6-blogDeep8/train6-blogDeep8_S00243_pred_tmax_sep.H > tBlogDeep8.H
	# Window3d n3=1 f3=$* < models/train6-blogDeep9/train6-blogDeep9_S00243_pred_tmax_sep.H > tBlogDeep9.H
	Window3d n3=1 f3=$* < models/train6-blogDeep10/train6-blogDeep10_S00243_pred_tmax_sep.H > tBlogDeep10.H
	Window3d n3=1 f3=$* < models/train6-blogDeep11/train6-blogDeep11_S00243_pred_tmax_sep.H > tBlogDeep11.H
	Window3d n3=1 f3=$* < models/train6-blogDeep12/train6-blogDeep12_S00243_pred_tmax_sep.H > tBlogDeep12.H
	Window3d n3=1 f3=$* < models/train6-blogDeep13/train6-blogDeep13_S00243_pred_tmax_sep.H > tBlogDeep13.H
	Window3d n3=1 f3=$* < models/train6-blogDeep14/train6-blogDeep14_S00243_pred_tmax_sep.H > tBlogDeep14.H
	Window3d n3=1 f3=$* < models/train6-blogDeep15/train6-blogDeep15_S00243_pred_tmax_sep.H > tBlogDeep15.H
	Window3d n3=1 f3=$* < models/train6-blogDeep16/train6-blogDeep16_S00243_pred_tmax_sep.H > tBlogDeep16.H
	Window3d n3=1 f3=$* < models/train6-blogDeep17/train6-blogDeep17_S00243_pred_tmax_sep.H > tBlogDeep17.H
	# Cat axis=3 tTrain.H tBlogDeep3.H tTrain.H tBlogDeep4.H tTrain.H tBlogDeep5.H tTrain.H tBlogDeep6.H | Grey color=j newclip=1 bclip=${bclip6} eclip=${eclip6} grid=y titles="Train:Deep3:Train:Deep4:Train:Deep5:Train:Deep6" gainpanel=a | Xtpen pixmaps=y &
	Cat axis=3 tTrain.H tBlogDeep10.H tTrain.H tBlogDeep11.H tTrain.H tBlogDeep12.H tTrain.H tBlogDeep13.H tTrain.H tBlogDeep14.H tTrain.H tBlogDeep15.H tTrain.H tBlogDeep16.H tTrain.H tBlogDeep17.H | Grey color=j newclip=1 bclip=${bclip6} eclip=${eclip6} grid=y titles="Train:Deep10:Train:Deep11:Train:Deep12:Train:Deep13:Train:Deep14:Train:Deep15:Train:Deep16:Train:Deep17" gainpanel=a | Xtpen pixmaps=y &
	# Cat axis=3 tTrain.H tBlogDeep10.H tTrain.H tBlogDeep11.H tTrain.H tBlogDeep12.H tTrain.H tBlogDeep13.H tTrain.H tBlogDeep17.H | Grey color=j newclip=1 bclip=${bclip6} eclip=${eclip6} grid=y titles="Train:Deep10:Train:Deep11:Train:Deep12:Train:Deep13:Train:Deep17" gainpanel=a | Xtpen pixmaps=y &
