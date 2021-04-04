################################################################################
################################# Training subset 2 ############################
################################################################################
# Training/dev/test on one full slice
n_epoch3=350
train3_file=par/train_file3.txt
dev3_file=par/dev_file3.txt

################################## Baseline ####################################
train3-win1-base1:
	rm -rf models/train3-win1-base1
	python ./python/CTP_main.py train train3-win1-base1 --model baseline --n_epochs 100 --lr=0.1 --device cuda:0 --train_file_list ${train3_file} --dev_file_list ${dev3_file} --baseline_n_hidden 100

train3-win1-base2:
	rm -rf models/train3-win1-base2
	python ./python/CTP_main.py train train3-win1-base2 --model baseline --n_epochs 100 --lr=0.01 --device cuda:0 --train_file_list ${train3_file} --dev_file_list ${dev3_file} --baseline_n_hidden 100

train3-win1-base3:
	rm -rf models/train3-win1-base3
	python ./python/CTP_main.py train train3-win1-base3 --model baseline --n_epochs 100 --lr=0.001 --device cuda:0 --train_file_list ${train3_file} --dev_file_list ${dev3_file} --baseline_n_hidden 100

makeBase3:
	# make train3-win1-base1 -B
	# make train3-win1-base2 -B
	make train3-win1-base3 -B

################################## FC Deep #####################################
train3-win1-fcDeep1:
	rm -rf models/train3-win1-fcDeep1
	python ./python/CTP_main.py train train3-win1-fcDeep1 --model fc6 --n_epochs 1000 --lr=0.001 --device cuda:1 --train_file_list ${train3_file} --dev_file_list ${dev3_file}

train3-win1-fcDeep2:
	rm -rf models/train3-win1-fcDeep2
	python ./python/CTP_main.py train train3-win1-fcDeep2 --model fc6 --n_epochs 1000 --lr=0.01 --device cuda:2 --train_file_list ${train3_file} --dev_file_list ${dev3_file}

train3-win1-fcDeep3:
	rm -rf models/train3-win1-fcDeep3
	python ./python/CTP_main.py train train3-win1-fcDeep3 --model fc6 --n_epochs 100 --lr=0.1 --device cuda:3 --train_file_list ${train3_file} --dev_file_list ${dev3_file}

train3-win1-fcDeep4:
	rm -rf models/train3-win1-fcDeep4
	python ./python/CTP_main.py train train3-win1-fcDeep4 --model fc6 --n_epochs 300 --lr=0.001 --lr_decay decay --decay_rate 0.0005 --device cuda:3 --train_file_list ${train3_file} --dev_file_list ${dev3_file}

train3-win1-fcDeep5:
	rm -rf models/train3-win1-fcDeep5
	python ./python/CTP_main.py train train3-win1-fcDeep5 --model fc6 --n_epochs 300 --lr=0.01 --lr_decay decay --decay_rate 0.0005 --device cuda:1 --train_file_list ${train3_file} --dev_file_list ${dev3_file}

train3-win1-fcDeep6:
	rm -rf models/train3-win1-fcDeep6
	python ./python/CTP_main.py train train3-win1-fcDeep6 --model fc6 --n_epochs 300 --lr=0.001 --lr_decay decay --decay_rate 0.0005 --loss mae --device cuda:0 --train_file_list ${train3_file} --dev_file_list ${dev3_file}

train3-win1-fcDeep7:
	rm -rf models/train3-win1-fcDeep7
	python ./python/CTP_main.py train train3-win1-fcDeep7 --model fc6 --n_epochs 300 --lr=0.001 --lr_decay decay --decay_rate 0.0005 --loss huber --device cuda:1 --train_file_list ${train3_file} --dev_file_list ${dev3_file}

train3-win1-fcDeep8:
	rm -rf models/train3-win1-fcDeep8
	python ./python/CTP_main.py train train3-win1-fcDeep8 --model fc6 --n_epochs 200 --lr=0.001 --device cuda:1 --train_file_list ${train3_file} --dev_file_list ${dev3_file} --batch_size=-1

makeFcDeep3:
	make train2-win1-fcDeep1 -B

################################### Kir ########################################
train3-win1-kir1:
	rm -rf models/train3-win1-kir1
	python ./python/CTP_main.py train train3-win1-kir1 --model kiranyaz --n_epochs 100 --lr=0.001 --device cuda:0 --train_file_list ${train3_file} --dev_file_list ${dev3_file}

train3-win1-kir2:
	rm -rf models/train3-win1-kir2
	python ./python/CTP_main.py train train3-win1-kir2 --model kiranyaz --n_epochs 100 --lr=0.01 --device cuda:1 --train_file_list ${train3_file} --dev_file_list ${dev3_file}

train3-win1-kir3:
	rm -rf models/train3-win1-kir3
	python ./python/CTP_main.py train train3-win1-kir3 --model kiranyaz --n_epochs 100 --lr=0.1 --device cuda:2 --train_file_list ${train3_file} --dev_file_list ${dev3_file}

train3-win1-kir4:
	rm -rf models/train3-win1-kir4
	python ./python/CTP_main.py train train3-win1-kir4 --model kiranyaz --n_epochs 100 --lr=0.001 --lr_decay decay --decay_rate 0.0005 --device cuda:0 --train_file_list ${train3_file} --dev_file_list ${dev3_file}

train3-win1-kir5:
	rm -rf models/train3-win1-kir5
	python ./python/CTP_main.py train train3-win1-kir5 --model kiranyaz --n_epochs 100 --lr=0.01 --lr_decay decay --decay_rate 0.0005 --device cuda:1 --train_file_list ${train3_file} --dev_file_list ${dev3_file}

train3-win1-kir6:
	rm -rf models/train3-win1-kir6
	python ./python/CTP_main.py train train3-win1-kir6 --model kiranyaz --n_epochs 100 --lr=0.001 --lr_decay exp --decay_gamma 0.995 --device cuda:0 --train_file_list ${train3_file} --dev_file_list ${dev3_file}

train3-win1-kir7:
	rm -rf models/train3-win1-kir7
	python ./python/CTP_main.py train train3-win1-kir7 --model kiranyaz --n_epochs 100 --lr=0.01 --lr_decay exp --decay_gamma 0.995 --device cuda:1 --train_file_list ${train3_file} --dev_file_list ${dev3_file}

train3-win1-kir8:
	rm -rf models/train3-win1-kir8
	python ./python/CTP_main.py train train3-win1-kir8 --model kiranyaz --n_epochs 100 --lr=0.001 --lr_decay step --step_size 40 --decay_gamma 0.99 --device cuda:2 --train_file_list ${train3_file} --dev_file_list ${dev3_file}

train3-win1-kir9:
	rm -rf models/train3-win1-kir9
	python ./python/CTP_main.py train train3-win1-kir9 --model kiranyaz --n_epochs 100 --lr=0.01 --lr_decay step --step_size 40 --decay_gamma 0.99 --device cuda:3 --train_file_list ${train3_file} --dev_file_list ${dev3_file}

train3-win1-kir10:
	rm -rf models/train3-win1-kir10
	python ./python/CTP_main.py train train3-win1-kir10 --model kiranyaz --n_epochs 100 --lr=0.005 --lr_decay step --step_size 40 --decay_gamma 0.995 --device cuda:2 --train_file_list ${train3_file} --dev_file_list ${dev3_file}

train3-win1-kir11:
	rm -rf models/train3-win1-kir11
	python ./python/CTP_main.py train train3-win1-kir11 --model kiranyaz --n_epochs 200 --lr=0.002 --lr_decay step --step_size 40 --decay_gamma 0.995 --device cuda:0 --train_file_list ${train3_file} --dev_file_list ${dev3_file}

train3-win1-kir12:
	rm -rf models/train3-win1-kir12
	python ./python/CTP_main.py train train3-win1-kir12 --model kiranyaz --n_epochs 200 --lr=0.002 --lr_decay step --step_size 20 --decay_gamma 0.999 --device cuda:1 --train_file_list ${train3_file} --dev_file_list ${dev3_file}

train3-win1-kir13:
	rm -rf models/train3-win1-kir13
	python ./python/CTP_main.py train train3-win1-kir13 --model kiranyaz --n_epochs 1000 --lr=0.004 --lr_decay step --step_size 20 --decay_gamma 0.999 --device cuda:0 --train_file_list ${train3_file} --dev_file_list ${dev3_file}

train3-win1-kir14:
	rm -rf models/train3-win1-kir14
	python ./python/CTP_main.py train train3-win1-kir14 --model kiranyaz --n_epochs 300 --lr=0.001 --device cuda:1 --loss huber --train_file_list ${train3_file} --dev_file_list ${dev3_file}

train3-win1-kir15:
	rm -rf models/train3-win1-kir15
	python ./python/CTP_main.py train train3-win1-kir15 --model kiranyaz --n_epochs 300 --lr=0.001 --device cuda:2 --loss mae --train_file_list ${train3_file} --dev_file_list ${dev3_file}

makeKir3:
	make train3-win1-kir6 -B

################################### Blog Med ###################################
# MSE
train3-win1-blogMed1:
	rm -rf models/train3-win1-blogMed1
	python ./python/CTP_main.py train train3-win1-blogMed1 --model blogMed --n_epochs 100 --lr=0.001 --device cuda:1 --train_file_list ${train3_file} --dev_file_list ${dev3_file}

train3-win1-blogMed2:
	rm -rf models/train3-win1-blogMed2
	python ./python/CTP_main.py train train3-win1-blogMed2 --model blogMed --n_epochs 100 --lr=0.01 --device cuda:2 --train_file_list ${train3_file} --dev_file_list ${dev3_file}

train3-win1-blogMed3:
	rm -rf models/train3-win1-blogMed3
	python ./python/CTP_main.py train train3-win1-blogMed3 --model blogMed --n_epochs 100 --lr=0.1 --device cuda:3 --train_file_list ${train3_file} --dev_file_list ${dev3_file}

train3-win1-blogMed4:
	rm -rf models/train3-win1-blogMed4
	python ./python/CTP_main.py train train3-win1-blogMed4 --model blogMed --n_epochs 100 --lr=0.005 --device cuda:1 --train_file_list ${train3_file} --dev_file_list ${dev3_file}

train3-win1-blogMed5:
	rm -rf models/train3-win1-blogMed5
	python ./python/CTP_main.py train train3-win1-blogMed5 --model blogMed --n_epochs 200 --lr=0.001 --lr_decay step --step_size 20 --decay_gamma 0.99 --device cuda:1 --train_file_list ${train3_file} --dev_file_list ${dev3_file}

train3-win1-blogMed6:
	rm -rf models/train3-win1-blogMed6
	python ./python/CTP_main.py train train3-win1-blogMed6 --model blogMed --n_epochs 200 --lr=0.001 --lr_decay exp --decay_gamma 0.99 --device cuda:0 --train_file_list ${train3_file} --dev_file_list ${dev3_file}

train3-win1-blogMed7:
	rm -rf models/train3-win1-blogMed7
	python ./python/CTP_main.py train train3-win1-blogMed7 --model blogMed --n_epochs 200 --lr=0.001 --lr_decay decay --decay_rate 0.0005 --device cuda:2 --train_file_list ${train3_file} --dev_file_list ${dev3_file}

train3-win1-blogMed8:
	rm -rf models/train3-win1-blogMed8
	python ./python/CTP_main.py train train3-win1-blogMed8 --model blogMed --n_epochs 100 --lr=0.001 --device cuda:3 --train_file_list ${train3_file} --dev_file_list ${dev3_file} --batch_size 4096

train3-win1-blogMed9:
	rm -rf models/train3-win1-blogMed9
	python ./python/CTP_main.py train train3-win1-blogMed9 --model blogMed --n_epochs 100 --lr=0.001 --lr=0.001 --lr_decay decay --decay_rate 0.0005 --device cuda:3 --train_file_list ${train3_file} --dev_file_list ${dev3_file} --batch_size 4096

train3-win1-blogMed10:
	rm -rf models/train3-win1-blogMed10
	python ./python/CTP_main.py train train3-win1-blogMed10 --model blogMed --n_epochs 100 --lr=0.01 --lr_decay decay --decay_rate 0.0005 --device cuda:3 --train_file_list ${train3_file} --dev_file_list ${dev3_file} --batch_size 4096

train3-win1-blogMed11:
	rm -rf models/train3-win1-blogMed11
	python ./python/CTP_main.py train train3-win1-blogMed11 --model blogMed --n_epochs 500 --lr=0.03 --lr_decay decay --decay_rate 0.0005 --device cuda:3 --train_file_list ${train3_file} --dev_file_list ${dev3_file} --batch_size 4096

train3-win1-blogMed12:
	rm -rf models/train3-win1-blogMed12
	python ./python/CTP_main.py train train3-win1-blogMed12 --model blogMed --n_epochs 500 --lr=0.03 --device cuda:3 --train_file_list ${train3_file} --dev_file_list ${dev3_file} --batch_size 4096

# Huber
train3-win1-blogMed13:
	rm -rf models/train3-win1-blogMed13
	python ./python/CTP_main.py train train3-win1-blogMed13 --model blogMed --n_epochs 100 --lr=0.001 --loss huber --device cuda:1 --train_file_list ${train3_file} --dev_file_list ${dev3_file} --batch_size 4096

train3-win1-blogMed14:
	rm -rf models/train3-win1-blogMed14
	python ./python/CTP_main.py train train3-win1-blogMed14 --model blogMed --n_epochs 500 --lr=0.01 --loss huber --device cuda:1 --train_file_list ${train3_file} --dev_file_list ${dev3_file} --batch_size 4096

train3-win1-blogMed15:
	rm -rf models/train3-win1-blogMed15
	python ./python/CTP_main.py train train3-win1-blogMed15 --model blogMed --n_epochs 500 --lr=0.01 --lr_decay decay --decay_rate 0.0005 --loss huber --device cuda:2 --train_file_list ${train3_file} --dev_file_list ${dev3_file} --batch_size 4096

train3-win1-blogMed16:
	rm -rf models/train3-win1-blogMed16
	python ./python/CTP_main.py train train3-win1-blogMed16 --model blogMed --n_epochs 500 --lr=0.01 --lr_decay step --step_size 20 --decay_gamma 0.99 --loss huber --device cuda:1 --train_file_list ${train3_file} --dev_file_list ${dev3_file} --batch_size 4096

train3-win1-blogMed17:
	rm -rf models/train3-win1-blogMed17
	python ./python/CTP_main.py train train3-win1-blogMed17 --model blogMed --n_epochs 500 --lr=0.001 --lr_decay step --step_size 20 --decay_gamma 0.99 --loss huber --device cuda:0 --train_file_list ${train3_file} --dev_file_list ${dev3_file} --batch_size 4096

train3-win1-blogMed18:
	rm -rf models/train3-win1-blogMed18
	python ./python/CTP_main.py train train3-win1-blogMed18 --model blogMed --n_epochs 300 --lr=0.0001 --device cuda:1 --train_file_list ${train3_file} --dev_file_list ${dev3_file} --batch_size 1024

train3-win1-blogMed19:
	rm -rf models/train3-win1-blogMed19
	python ./python/CTP_main.py train train3-win1-blogMed19 --model blogMed --n_epochs 500 --lr=0.01 --loss mae --device cuda:1 --train_file_list ${train3_file} --dev_file_list ${dev3_file} --batch_size 4096

makeBlogMed3:
	make train3-win1-blogMed1 -B

################################### Blog Deep ###################################
train3-win1-blogDeep1:
	rm -rf models/train3-win1-blogDeep1
	python ./python/CTP_main.py train train3-win1-blogDeep1 --model blogDeep --n_epochs 100 --lr=0.001 --device cuda:3 --train_file_list ${train3_file} --dev_file_list ${dev3_file}

train3-win1-blogDeep2:
	rm -rf models/train3-win1-blogDeep2
	python ./python/CTP_main.py train train3-win1-blogDeep2 --model blogDeep --n_epochs 100 --lr=0.0001 --device cuda:0 --train_file_list ${train3_file} --dev_file_list ${dev3_file}

train3-win1-blogDeep3:
	rm -rf models/train3-win1-blogDeep3
	python ./python/CTP_main.py train train3-win1-blogDeep3 --model blogDeep --n_epochs 100 --lr=0.01 --device cuda:2 --train_file_list ${train3_file} --dev_file_list ${dev3_file}

train3-win1-blogDeep4:
	rm -rf models/train3-win1-blogDeep4
	python ./python/CTP_main.py train train3-win1-blogDeep4 --model blogDeep --n_epochs 100 --lr=0.001 --device cuda:0 --train_file_list ${train3_file} --dev_file_list ${dev3_file} --batch_size 2048

train3-win1-blogDeep5:
	rm -rf models/train3-win1-blogDeep5
	python ./python/CTP_main.py train train3-win1-blogDeep5 --model blogDeep --n_epochs 100 --lr=0.001 --device cuda:1 --train_file_list ${train3_file} --dev_file_list ${dev3_file} --batch_size 1024

train3-win1-blogDeep6:
	rm -rf models/train3-win1-blogDeep6
	python ./python/CTP_main.py train train3-win1-blogDeep6 --model blogDeep --n_epochs 100 --lr=0.005 --device cuda:2 --train_file_list ${train3_file} --dev_file_list ${dev3_file}

train3-win1-blogDeep7:
	rm -rf models/train3-win1-blogDeep7
	python ./python/CTP_main.py train train3-win1-blogDeep7 --model blogDeep --n_epochs 100 --lr=0.0005 --device cuda:3 --train_file_list ${train3_file} --dev_file_list ${dev3_file}

train3-win1-blogDeep8:
	rm -rf models/train3-win1-blogDeep8
	python ./python/CTP_main.py train train3-win1-blogDeep8 --model blogDeep --n_epochs 500 --lr=0.005 --lr_decay step --step_size 20 --decay_gamma 0.99 --device cuda:2 --train_file_list ${train3_file} --dev_file_list ${dev3_file} --batch_size 1024

train3-win1-blogDeep9:
	rm -rf models/train3-win1-blogDeep9
	python ./python/CTP_main.py train train3-win1-blogDeep9 --model blogDeep --n_epochs 500 --lr=0.005 --lr_decay exp --decay_gamma 0.995 --device cuda:0 --train_file_list ${train3_file} --dev_file_list ${dev3_file} --batch_size 1024

train3-win1-blogDeep10:
	rm -rf models/train3-win1-blogDeep10
	python ./python/CTP_main.py train train3-win1-blogDeep10 --model blogDeep --n_epochs 500 --lr_decay decay --decay_rate 0.0005 --device cuda:1 --train_file_list ${train3_file} --dev_file_list ${dev3_file} --batch_size 1024

makeBlogDeep3:
	make train3-win1-blogDeep1 -B

################################################################################
############################ Prediction on train data ##########################
################################################################################
pred3_file_train=dat/S00243-train3-win2.h5

predictTrain3-win1-base%:
	python ./python/CTP_main.py predict train3-win1-base$* --model baseline --patient_file ${pred3_file_train} --patient_id S00243 --baseline_n_hidden 100

predictTrain3-win1-fcDeep%:
	python ./python/CTP_main.py predict train3-win1-fcDeep$* --model fc6 --device cuda:0 --patient_file ${pred3_file_train} --patient_id S00243

predictTrain3-win1-kir%:
	python ./python/CTP_main.py predict train3-win1-kir$* --model kiranyaz --device cuda:0 --patient_file ${pred3_file_train} --patient_id S00243

predictTrain3-win1-blogMed%:
	python ./python/CTP_main.py predict train3-win1-blogMed$* --model blogMed --device cuda:0 --patient_file ${pred3_file_train} --patient_id S00243

predictTrain3-win1-blogDeep%:
	python ./python/CTP_main.py predict train3-win1-blogDeep$* --model blogDeep --device cuda:0 --patient_file ${pred3_file_train} --patient_id S00243

makePredTrain3:
	# make predictTrain3-win1-base3 -B
	# make predictTrain3-win1-fcDeep1 -B
	# make predictTrain3-win1-fcDeep2 -B
	# make predictTrain3-win1-fcDeep1 -B
	# make predictTrain3-win1-fcDeep4 -B
	# make predictTrain3-win1-fcDeep5 -B
	# make predictTrain3-win1-fcDeep6 -B
	# make predictTrain3-win1-fcDeep7 -B
	# make predictTrain3-win1-fcDeep8 -B
	# make predictTrain3-win1-kir13 -B
	# make predictTrain3-win1-kir14 -B
	# make predictTrain3-win1-kir15 -B
	# make predictTrain3-win1-blogMed1 -B
	# make predictTrain3-win1-blogMed4 -B
	# make predictTrain3-win1-blogMed5 -B
	# make predictTrain3-win1-blogMed8 -B
	# make predictTrain3-win1-blogMed13 -B
	# make predictTrain3-win1-blogMed14 -B
	# make predictTrain3-win1-blogMed19 -B
	make predictTrain3-win1-blogDeep1 -B
	make predictTrain3-win1-blogDeep4 -B
	make predictTrain3-win1-blogDeep8 -B
	make predictTrain3-win1-blogDeep9 -B

################################################################################
################################# Display ######################################
################################################################################
dispResult3-s%:
	# Training data
	Window3d n3=1 f3=$* < ${pred3_file_train}_tmax_m.H > t1.H
	Window3d n3=1 f3=$* < models/train3-win1-base3/train3-win1-base3_S00243_pred_tmax_sep.H > t2.H
	Window3d n3=1 f3=$* < models/train3-win1-fcDeep1/train3-win1-fcDeep1_S00243_pred_tmax_sep.H > t3a.H
	Window3d n3=1 f3=$* < models/train3-win1-fcDeep4/train3-win1-fcDeep4_S00243_pred_tmax_sep.H > t3b.H
	Window3d n3=1 f3=$* < models/train3-win1-fcDeep5/train3-win1-fcDeep5_S00243_pred_tmax_sep.H > t3c.H
	Window3d n3=1 f3=$* < models/train3-win1-fcDeep6/train3-win1-fcDeep6_S00243_pred_tmax_sep.H > t3d.H
	Window3d n3=1 f3=$* < models/train3-win1-fcDeep7/train3-win1-fcDeep7_S00243_pred_tmax_sep.H > t3e.H
	Window3d n3=1 f3=$* < models/train3-win1-fcDeep8/train3-win1-fcDeep8_S00243_pred_tmax_sep.H > t3f.H
	# Window3d n3=1 f3=$* < models/train3-win1-kir13/train3-win1-kir13_S00243_pred_tmax_sep.H > t4a.H
	# Window3d n3=1 f3=$* < models/train3-win1-kir14/train3-win1-kir14_S00243_pred_tmax_sep.H > t4b.H
	# Window3d n3=1 f3=$* < models/train3-win1-kir15/train3-win1-kir15_S00243_pred_tmax_sep.H > t4c.H
	# Window3d n3=1 f3=$* < models/train3-win1-blogMed1/train3-win1-blogMed1_S00243_pred_tmax_sep.H > t5a.H
	# Window3d n3=1 f3=$* < models/train3-win1-blogMed4/train3-win1-blogMed4_S00243_pred_tmax_sep.H > t5b.H
	# Window3d n3=1 f3=$* < models/train3-win1-blogMed5/train3-win1-blogMed5_S00243_pred_tmax_sep.H > t5c.H
	# Window3d n3=1 f3=$* < models/train3-win1-blogMed8/train3-win1-blogMed8_S00243_pred_tmax_sep.H > t5d.H
	# Window3d n3=1 f3=$* < models/train3-win1-blogMed13/train3-win1-blogMed13_S00243_pred_tmax_sep.H > t5e.H
	# Window3d n3=1 f3=$* < models/train3-win1-blogMed14/train3-win1-blogMed14_S00243_pred_tmax_sep.H > t5f.H
	# Window3d n3=1 f3=$* < models/train3-win1-blogMed19/train3-win1-blogMed19_S00243_pred_tmax_sep.H > t5g.H
	# Window3d n3=1 f3=$* < models/train3-win1-blogDeep1/train3-win1-blogDeep1_S00243_pred_tmax_sep.H > t6a.H
	# Window3d n3=1 f3=$* < models/train3-win1-blogDeep4/train3-win1-blogDeep4_S00243_pred_tmax_sep.H > t6b.H
	# Window3d n3=1 f3=$* < models/train3-win1-blogDeep8/train3-win1-blogDeep8_S00243_pred_tmax_sep.H > t6c.H
	# Window3d n3=1 f3=$* < models/train3-win1-blogDeep9/train3-win1-blogDeep9_S00243_pred_tmax_sep.H > t6d.H
	Cat axis=3 t1.H t2.H t1.H t3a.H t1.H t3b.H t1.H t3c.H t1.H t3d.H t1.H t3e.H t1.H t3f.H | Grey color=j newclip=1 bclip=-10 eclip=160 grid=y titles="True:Base:True:Fc1:True:Fc4:True:Fc5:True:Fc6:True:Fc7:True:Fc8" gainpanel=a | Xtpen pixmaps=y &
	# Cat axis=3 t1.H t4a.H t1.H t4b.H t1.H t4c.H | Grey color=j newclip=1 bclip=0 eclip=160 grid=y titles="True:Kir13:True:Kir14:True:Kir15" gainpanel=a | Xtpen pixmaps=y &
	# Cat axis=3 t1.H t5a.H t1.H t5b.H t1.H t5c.H t1.H t5d.H t1.H t5e.H t1.H t5f.H t1.H t5g.H | Grey color=j newclip=1 bclip=0 eclip=160 grid=y titles="True:Med1:True:Med4:True:Med5:True:Med8:True:Med13:True:Med14:True:Med19" gainpanel=a | Xtpen pixmaps=y &
	# Cat axis=3 t1.H t6a.H t1.H t6b.H t1.H t6c.H t1.H t6d.H | Grey color=j newclip=1 bclip=0 eclip=160 grid=y titles="True:Deep1:True:Deep4:True:Deep8:True:Deep9" gainpanel=a | Xtpen pixmaps=y &

################################################################################
############################ Prediction on dev data ############################
################################################################################
pred3_file_dev=dat/S00243-train3-win3.h5
# pred3_file_dev=dat/S00295-train3-win4.h5

predictDev3-win1-base%:
	python ./python/CTP_main.py predict train3-win1-base$* --model baseline --patient_file ${pred3_file_dev} --patient_id S00243d --baseline_n_hidden 100

predictDev3-win1-fcDeep%:
	python ./python/CTP_main.py predict train3-win1-fcDeep$* --model fc6 --device cuda:0 --patient_file ${pred3_file_dev} --patient_id S00243d

predictDev3-win1-kir%:
	python ./python/CTP_main.py predict train3-win1-kir$* --model kiranyaz --device cuda:0 --patient_file ${pred3_file_dev} --patient_id S00243d

predictDev3-win1-blogMed%:
	python ./python/CTP_main.py predict train3-win1-blogMed$* --model blogMed --device cuda:0 --patient_file ${pred3_file_dev} --patient_id S00243d

predictDev3-win1-blogDeep%:
	python ./python/CTP_main.py predict train3-win1-blogDeep$* --model blogDeep --device cuda:0 --patient_file ${pred3_file_dev} --patient_id S00243d

makePredDev3:
	# make predictDev3-win1-base3 -B
	# make predictDev3-win1-fcDeep1 -B
	# make predictDev3-win1-fcDeep2 -B
	# make predictDev3-win1-fcDeep1 -B
	# make predictDev3-win1-fcDeep4 -B
	# make predictDev3-win1-fcDeep5 -B
	# make predictDev3-win1-fcDeep6 -B
	# make predictDev3-win1-fcDeep7 -B
	# make predictDev3-win1-fcDeep8 -B
	# make predictDev3-win1-kir3 -B
	# make predictDev3-win1-kir13 -B
	# make predictDev3-win1-kir14 -B
	# make predictDev3-win1-kir15 -B
	# make predictDev3-win1-blogMed1 -B
	# make predictDev3-win1-blogMed4 -B
	# make predictDev3-win1-blogMed5 -B
	# make predictDev3-win1-blogMed8 -B
	# make predictDev3-win1-blogMed13 -B
	# make predictDev3-win1-blogMed14 -B
	# make predictDev3-win1-blogMed19 -B
	make predictDev3-win1-blogDeep1 -B
	make predictDev3-win1-blogDeep4 -B
	make predictDev3-win1-blogDeep8 -B
	make predictDev3-win1-blogDeep9 -B

################################################################################
################################# Display ######################################
################################################################################
dispResult3Dev-s%:
	# Training data
	Window3d n3=1 f3=$* < ${pred3_file_dev}_tmax_m.H > t1.H
	# Window3d n3=1 f3=$* < models/train3-win1-base3/train3-win1-base3_S00243d_pred_tmax_sep.H > t2.H
	# Window3d n3=1 f3=$* < models/train3-win1-fcDeep1/train3-win1-fcDeep1_S00243d_pred_tmax_sep.H > t3a.H
	# Window3d n3=1 f3=$* < models/train3-win1-fcDeep4/train3-win1-fcDeep4_S00243d_pred_tmax_sep.H > t3b.H
	# Window3d n3=1 f3=$* < models/train3-win1-fcDeep5/train3-win1-fcDeep5_S00243d_pred_tmax_sep.H > t3c.H
	# Window3d n3=1 f3=$* < models/train3-win1-fcDeep6/train3-win1-fcDeep6_S00243d_pred_tmax_sep.H > t3d.H
	# Window3d n3=1 f3=$* < models/train3-win1-fcDeep7/train3-win1-fcDeep7_S00243d_pred_tmax_sep.H > t3e.H
	# Window3d n3=1 f3=$* < models/train3-win1-fcDeep8/train3-win1-fcDeep8_S00243d_pred_tmax_sep.H > t3f.H
	# Window3d n3=1 f3=$* < models/train3-win1-kir13/train3-win1-kir13_S00243d_pred_tmax_sep.H > t4a.H
	# Window3d n3=1 f3=$* < models/train3-win1-kir14/train3-win1-kir14_S00243d_pred_tmax_sep.H > t4b.H
	# Window3d n3=1 f3=$* < models/train3-win1-kir15/train3-win1-kir15_S00243d_pred_tmax_sep.H > t4c.H
	# Window3d n3=1 f3=$* < models/train3-win1-blogMed1/train3-win1-blogMed1_S00243d_pred_tmax_sep.H > t5a.H
	# Window3d n3=1 f3=$* < models/train3-win1-blogMed4/train3-win1-blogMed4_S00243d_pred_tmax_sep.H > t5b.H
	# Window3d n3=1 f3=$* < models/train3-win1-blogMed5/train3-win1-blogMed5_S00243d_pred_tmax_sep.H > t5c.H
	# Window3d n3=1 f3=$* < models/train3-win1-blogMed8/train3-win1-blogMed8_S00243d_pred_tmax_sep.H > t5d.H
	# Window3d n3=1 f3=$* < models/train3-win1-blogMed13/train3-win1-blogMed13_S00243d_pred_tmax_sep.H > t5e.H
	# Window3d n3=1 f3=$* < models/train3-win1-blogMed14/train3-win1-blogMed14_S00243d_pred_tmax_sep.H > t5f.H
	# Window3d n3=1 f3=$* < models/train3-win1-blogMed19/train3-win1-blogMed19_S00243d_pred_tmax_sep.H > t5g.H
	Window3d n3=1 f3=$* < models/train3-win1-blogDeep1/train3-win1-blogDeep1_S00243d_pred_tmax_sep.H > t6a.H
	Window3d n3=1 f3=$* < models/train3-win1-blogDeep4/train3-win1-blogDeep4_S00243d_pred_tmax_sep.H > t6b.H
	Window3d n3=1 f3=$* < models/train3-win1-blogDeep8/train3-win1-blogDeep8_S00243d_pred_tmax_sep.H > t6c.H
	Window3d n3=1 f3=$* < models/train3-win1-blogDeep9/train3-win1-blogDeep9_S00243d_pred_tmax_sep.H > t6d.H
	# Cat axis=3 t1.H t3a.H t1.H t3b.H t1.H t3c.H t1.H t3d.H t1.H t3e.H t1.H t3f.H | Grey color=j newclip=1 bclip=-10 eclip=160 grid=y titles="True:Fc1:True:Fc4:True:Fc5:True:Fc6:True:Fc7:True:Fc8" gainpanel=a | Xtpen pixmaps=y &
	# Cat axis=3 t1.H t4a.H t1.H t4b.H t1.H t4c.H | Grey color=j newclip=1 bclip=-10 eclip=160 grid=y titles="True:Kir13:True:Kir14:True:Kir15" gainpanel=a | Xtpen pixmaps=y &
	# Cat axis=3 t1.H t5a.H t1.H t5b.H t1.H t5c.H t1.H t5d.H t1.H t5e.H t1.H t5f.H t1.H t5g.H | Grey color=j newclip=1 bclip=0 eclip=160 grid=y titles="True:Med1:True:Med4:True:Med5:True:Med8:True:Med13:True:Med14:True:Med19" gainpanel=a | Xtpen pixmaps=y &
	Cat axis=3 t1.H t6a.H t1.H t6b.H t1.H t6c.H t1.H t6d.H | Grey color=j newclip=1 bclip=0 eclip=160 grid=y titles="True:Deep1:True:Deep4:True:Deep8:True:Deep9" gainpanel=a | Xtpen pixmaps=y &

################################################################################
############################ Prediction on test data ############################
################################################################################
pred3_file_test=dat/S00295-train3-win4.h5

predictTest3-win1-base%:
	python ./python/CTP_main.py predict train3-win1-base$* --model baseline --patient_file ${pred3_file_test} --patient_id S00243t --baseline_n_hidden 100

predictTest3-win1-fcDeep%:
	python ./python/CTP_main.py predict train3-win1-fcDeep$* --model fc6 --device cuda:0 --patient_file ${pred3_file_test} --patient_id S00243t

predictTest3-win1-kir%:
	python ./python/CTP_main.py predict train3-win1-kir$* --model kiranyaz --device cuda:0 --patient_file ${pred3_file_test} --patient_id S00243t

predictTest3-win1-blogMed%:
	python ./python/CTP_main.py predict train3-win1-blogMed$* --model blogMed --device cuda:0 --patient_file ${pred3_file_test} --patient_id S00243t

predictTest3-win1-blogDeep%:
	python ./python/CTP_main.py predict train3-win1-blogDeep$* --model blogDeep --device cuda:0 --patient_file ${pred3_file_test} --patient_id S00243t

makePredTest3:
	make predictTest3-win1-base3 -B
	make predictTest3-win1-fcDeep1 -B
	make predictTest3-win1-fcDeep2 -B
	make predictTest3-win1-fcDeep1 -B
	make predictTest3-win1-fcDeep4 -B
	make predictTest3-win1-fcDeep5 -B
	make predictTest3-win1-fcDeep6 -B
	make predictTest3-win1-fcDeep7 -B
	make predictTest3-win1-fcDeep8 -B
	make predictTest3-win1-kir3 -B
	make predictTest3-win1-kir13 -B
	make predictTest3-win1-kir14 -B
	make predictTest3-win1-kir15 -B
	make predictTest3-win1-blogMed1 -B
	make predictTest3-win1-blogMed4 -B
	make predictTest3-win1-blogMed5 -B
	make predictTest3-win1-blogMed8 -B
	make predictTest3-win1-blogMed13 -B
	make predictTest3-win1-blogMed14 -B
	make predictTest3-win1-blogMed19 -B
	make predictTest3-win1-blogDeep1 -B
	make predictTest3-win1-blogDeep4 -B
	make predictTest3-win1-blogDeep8 -B
	make predictTest3-win1-blogDeep9 -B
