################################################################################
################################# Training subset 2 ############################
################################################################################
# Training/dev/test on one full slice
train1_file=dat/S00243-train2-win2.h5
dev1_file=dat/S00243-train2-win3.h5
test1_file=dat/S00243-train2-win4.h5
n_epoch1=350

################################## Baseline ####################################
train-S00243-train2-win1-base1:
	rm -rf models/S00243-train2-win1-base1
	python ./python/CTP_main.py train S00243-train2-win1-base1 --model baseline --n_epochs 500 --lr=0.1 --device cuda:0 --train_file ${train1_file} --dev_file ${dev1_file} --baseline_n_hidden 100

train-S00243-train2-win1-base2:
	rm -rf models/S00243-train2-win1-base2
	python ./python/CTP_main.py train S00243-train2-win1-base2 --model baseline --n_epochs 500 --lr=0.01 --device cuda:0 --train_file ${train1_file} --dev_file ${dev1_file} --baseline_n_hidden 100

train-S00243-train2-win1-base3:
	rm -rf models/S00243-train2-win1-base3
	python ./python/CTP_main.py train S00243-train2-win1-base3 --model baseline --n_epochs 500 --lr=0.001 --device cuda:0 --train_file ${train1_file} --dev_file ${dev1_file} --baseline_n_hidden 100

# train-S00243-train2-win1-base4:
# 	rm -rf models/S00243-train2-win1-base4
# 	python ./python/CTP_main.py train S00243-train2-win1-base4 --model baseline --n_epochs 500 --lr=0.1 --lr_decay decay --decay_rate 0.0005 --device cuda:0 --train_file ${train1_file} --dev_file ${dev1_file} --baseline_n_hidden 100
#
# train-S00243-train2-win1-base5:
# 	rm -rf models/S00243-train2-win1-base5
# 	python ./python/CTP_main.py train S00243-train2-win1-base5 --model baseline --n_epochs 500 --lr=0.1 --device cuda:0 --train_file ${train1_file} --dev_file ${dev1_file} --baseline_n_hidden 100
#
# train-S00243-train2-win1-base6:
# 	rm -rf models/S00243-train2-win1-base6
# 	python ./python/CTP_main.py train S00243-train2-win1-base6 --model baseline --n_epochs 500 --lr=0.5 --device cuda:0 --train_file ${train1_file} --dev_file ${dev1_file} --baseline_n_hidden 100
#
# train-S00243-train2-win1-base7:
# 	rm -rf models/S00243-train2-win1-base7
# 	python ./python/CTP_main.py train S00243-train2-win1-base7 --model baseline --n_epochs 500 --lr=0.05 --lr_decay exp --decay_gamma 0.99 --device cuda:0 --train_file ${train1_file} --dev_file ${dev1_file} --baseline_n_hidden 100
#
# train-S00243-train2-win1-base8:
# 	rm -rf models/S00243-train2-win1-base8
# 	python ./python/CTP_main.py train S00243-train2-win1-base8 --model baseline --n_epochs 500 --lr=0.1 --lr_decay step --step_size 30 --decay_gamma 0.99 --device cuda:0 --train_file ${train1_file} --dev_file ${dev1_file} --baseline_n_hidden 100
#
# train-S00243-train2-win1-base9:
# 	rm -rf models/S00243-train2-win1-base9
# 	python ./python/CTP_main.py train S00243-train2-win1-base9 --model baseline --n_epochs 500 --lr=0.2 --lr_decay step --step_size 20 --decay_gamma 0.95 --device cuda:0 --train_file ${train1_file} --dev_file ${dev1_file} --baseline_n_hidden 100
#
# train-S00243-train2-win1-base10:
# 	rm -rf models/S00243-train2-win1-base10
# 	python ./python/CTP_main.py train S00243-train2-win1-base10 --model baseline --n_epochs 500 --lr=0.15 --lr_decay step --step_size 40 --decay_gamma 0.99 --device cuda:0 --train_file ${train1_file} --dev_file ${dev1_file} --baseline_n_hidden 100

makeBase:
	make train-S00243-train2-win1-base1 -B
	make train-S00243-train2-win1-base2 -B
	make train-S00243-train2-win1-base3 -B

################################## FC Deep #####################################
train-S00243-train2-win1-fcDeep1:
	rm -rf models/S00243-train2-win1-fcDeep1
	python ./python/CTP_main.py train S00243-train2-win1-fcDeep1 --model fc6 --n_epochs 1000 --lr=0.001 --lr_decay decay --device cuda:0 --train_file ${train1_file} --dev_file ${dev1_file}

train-S00243-train2-win1-fcDeep2:
	rm -rf models/S00243-train2-win1-fcDeep2
	python ./python/CTP_main.py train S00243-train2-win1-fcDeep2 --model fc6 --n_epochs 1000 --lr=0.001 --lr_decay decay --device cuda:0 --train_file ${train1_file} --dev_file ${dev1_file}	--loss mae

train-S00243-train2-win1-fcDeep3:
	rm -rf models/S00243-train2-win1-fcDeep3
	python ./python/CTP_main.py train S00243-train2-win1-fcDeep3 --model fc6 --n_epochs 1000 --lr=0.001 --lr_decay decay --device cuda:0 --train_file ${train1_file} --dev_file ${dev1_file}	--loss huber

train-S00243-train2-win1-fcDeep4:
	rm -rf models/S00243-train2-win1-fcDeep4
	python ./python/CTP_main.py train S00243-train2-win1-fcDeep4 --model fc6 --n_epochs ${n_epoch1} --lr=0.01 --lr_decay decay --device cuda:0 --train_file ${train1_file} --dev_file ${dev1_file}

train-S00243-train2-win1-fcDeep5:
	rm -rf models/S00243-train2-win1-fcDeep5
	python ./python/CTP_main.py train S00243-train2-win1-fcDeep5 --model fc6 --n_epochs ${n_epoch1} --lr=0.01 --lr_decay step --step_size 40 --decay_gamma 0.99 --device cuda:0 --train_file ${train1_file} --dev_file ${dev1_file}

train-S00243-train2-win1-fcDeep6:
	rm -rf models/S00243-train2-win1-fcDeep6
	python ./python/CTP_main.py train S00243-train2-win1-fcDeep6 --model fc6 --n_epochs 150 --lr=0.01 --lr_decay step --step_size 40 --decay_gamma 0.99 --device cuda:0 --train_file ${train1_file} --dev_file ${dev1_file}	--batch_size -1

train-S00243-train2-win1-fcDeep7:
	rm -rf models/S00243-train2-win1-fcDeep7
	python ./python/CTP_main.py train S00243-train2-win1-fcDeep7 --model fc6 --n_epochs 1000 --lr=0.01 --lr_decay step --step_size 40 --decay_gamma 0.99 --device cuda:0 --train_file ${train1_file} --dev_file ${dev1_file}

train-S00243-train2-win1-fcDeep8:
	rm -rf models/S00243-train2-win1-fcDeep8
	python ./python/CTP_main.py train S00243-train2-win1-fcDeep8 --model fc6 --n_epochs 500 --lr=0.01 --lr_decay step --step_size 40 --decay_gamma 0.99 --device cuda:0 --train_file ${train1_file} --dev_file ${dev1_file}

makeFcDeep:
	make train-S00243-train2-win1-fcDeep1 -B
	make train-S00243-train2-win1-fcDeep2 -B
	make train-S00243-train2-win1-fcDeep3 -B
	# make train-S00243-train2-win1-fcDeep4 -B
	# make train-S00243-train2-win1-fcDeep5 -B
	# make train-S00243-train2-win1-fcDeep6 -B
	# make train-S00243-train2-win1-fcDeep7 -B
	# make train-S00243-train2-win1-fcDeep8 -B

#################################### Kir #######################################
train-S00243-train2-win1-kir1:
	rm -rf models/S00243-train2-win1-kir1
	python ./python/CTP_main.py train S00243-train2-win1-kir1 --model kiranyaz --n_epochs ${n_epoch1} --lr=0.001 --lr_decay decay --device cuda:0 --train_file ${train1_file} --dev_file ${dev1_file}

train-S00243-train2-win1-kir2:
	rm -rf models/S00243-train2-win1-kir2
	python ./python/CTP_main.py train S00243-train2-win1-kir2 --model kiranyaz --n_epochs 1000 --lr=0.01 --lr_decay decay --device cuda:0 --train_file ${train1_file} --dev_file ${dev1_file}

train-S00243-train2-win1-kir3:
	rm -rf models/S00243-train2-win1-kir3
	python ./python/CTP_main.py train S00243-train2-win1-kir3 --model kiranyaz --n_epochs ${n_epoch1} --lr=0.05 --lr_decay decay --device cuda:0 --train_file ${train1_file} --dev_file ${dev1_file}

train-S00243-train2-win1-kir4:
	rm -rf models/S00243-train2-win1-kir4
	python ./python/CTP_main.py train S00243-train2-win1-kir4 --model kiranyaz --n_epochs ${n_epoch1} --lr=0.1 --lr_decay decay --device cuda:0 --train_file ${train1_file} --dev_file ${dev1_file}

train-S00243-train2-win1-kir5:
	rm -rf models/S00243-train2-win1-kir5
	python ./python/CTP_main.py train S00243-train2-win1-kir5 --model kiranyaz --n_epochs 1000 --lr=0.01 --lr_decay step --step_size 40 --decay_gamma 0.99 --device cuda:0 --train_file ${train1_file} --dev_file ${dev1_file}

train-S00243-train2-win1-kir6:
	rm -rf models/S00243-train2-win1-kir6
	python ./python/CTP_main.py train S00243-train2-win1-kir6 --model kiranyaz --n_epochs 1000 --lr=0.01 --device cuda:0 --train_file ${train1_file} --dev_file ${dev1_file} --batch_size 1024 --lr_decay step --step_size 10 --decay_gamma 0.99

train-S00243-train2-win1-kir7:
	rm -rf models/S00243-train2-win1-kir7
	python ./python/CTP_main.py train S00243-train2-win1-kir7 --model kiranyaz --n_epochs 500 --lr=0.005 --device cuda:0 --train_file ${train1_file} --dev_file ${dev1_file} --batch_size 1024

makeKir:
	# make train-S00243-train2-win1-kir1 -B
	# make train-S00243-train2-win1-kir2 -B
	# make train-S00243-train2-win1-kir3 -B
	# make train-S00243-train2-win1-kir4 -B
	make train-S00243-train2-win1-kir5 -B
	make train-S00243-train2-win1-kir6 -B
	# make train-S00243-train2-win1-kir7 -B

################################## Blog Med ####################################
train-S00243-train2-win1-blogMed1:
	rm -rf models/S00243-train2-win1-blogMed1
	python ./python/CTP_main.py train S00243-train2-win1-blogMed1 --model blogMed --n_epochs 1000 --lr=0.001 --lr_decay decay --device cuda:0 --train_file ${train1_file} --dev_file ${dev1_file}

train-S00243-train2-win1-blogMed2:
	rm -rf models/S00243-train2-win1-blogMed2
	python ./python/CTP_main.py train S00243-train2-win1-blogMed2 --model blogMed --n_epochs 1000 --lr=0.01 --lr_decay decay --device cuda:0 --train_file ${train1_file} --dev_file ${dev1_file}

train-S00243-train2-win1-blogMed3:
	rm -rf models/S00243-train2-win1-blogMed3
	python ./python/CTP_main.py train S00243-train2-win1-blogMed3 --model blogMed --n_epochs 1000 --lr=0.01 --lr_decay decay --lr_decay step --step_size 40 --decay_gamma 0.99 --device cuda:0 --train_file ${train1_file} --dev_file ${dev1_file}

train-S00243-train2-win1-blogMed4:
	rm -rf models/S00243-train2-win1-blogMed4
	python ./python/CTP_main.py train S00243-train2-win1-blogMed4 --model blogMed --n_epochs 1000 --lr=0.01 --lr_decay exp --decay_gamma 0.99 --device cuda:0 --train_file ${train1_file} --dev_file ${dev1_file}

train-S00243-train2-win1-blogMed5:
	rm -rf models/S00243-train2-win1-blogMed5
	python ./python/CTP_main.py train S00243-train2-win1-blogMed5 --model blogMed --n_epochs 1000 --lr=0.02 --lr_decay exp --decay_gamma 0.995 --device cuda:0 --train_file ${train1_file} --dev_file ${dev1_file}

train-S00243-train2-win1-blogMed6:
	rm -rf models/S00243-train2-win1-blogMed6
	python ./python/CTP_main.py train S00243-train2-win1-blogMed6 --model blogMed --n_epochs 1000 --lr=0.005 --device cuda:0 --train_file ${train1_file} --dev_file ${dev1_file}

train-S00243-train2-win1-blogMed7:
	rm -rf models/S00243-train2-win1-blogMed7
	python ./python/CTP_main.py train S00243-train2-win1-blogMed7 --model blogMed --n_epochs 1000 --lr=0.05 --lr_decay exp --decay_gamma 0.995 --device cuda:0 --train_file ${train1_file} --dev_file ${dev1_file}

makeMed:
	# make train-S00243-train2-win1-blogMed1 -B
	# make train-S00243-train2-win1-blogMed2 -B
	# make train-S00243-train2-win1-blogMed3 -B
	# make train-S00243-train2-win1-blogMed4 -B
	# make train-S00243-train2-win1-blogMed5 -B
	# make train-S00243-train2-win1-blogMed6 -B
	make train-S00243-train2-win1-blogMed7 -B

################################## Blog Deep ###################################
# Blog deep
train-S00243-train2-win1-blogDeep1:
	rm -rf models/S00243-train2-win1-blogDeep1
	python ./python/CTP_main.py train S00243-train2-win1-blogDeep1 --model blogDeep --n_epochs 1000 --lr=0.001 --lr_decay decay --device cuda:0 --train_file ${train1_file} --dev_file ${dev1_file}

train-S00243-train2-win1-blogDeep2:
	rm -rf models/S00243-train2-win1-blogDeep2
	python ./python/CTP_main.py train S00243-train2-win1-blogDeep2 --model blogDeep --n_epochs 500 --lr=0.01 --lr_decay decay --device cuda:0 --train_file ${train1_file} --dev_file ${dev1_file}

train-S00243-train2-win1-blogDeep3:
	rm -rf models/S00243-train2-win1-blogDeep3
	python ./python/CTP_main.py train S00243-train2-win1-blogDeep3 --model blogDeep --n_epochs 500 --lr=0.005 --lr_decay decay --device cuda:0 --train_file ${train1_file} --dev_file ${dev1_file}

train-S00243-train2-win1-blogDeep4:
	rm -rf models/S00243-train2-win1-blogDeep4
	python ./python/CTP_main.py train S00243-train2-win1-blogDeep4 --model blogDeep --n_epochs 500 --lr=0.005 --lr_decay decay --device cuda:0 --train_file ${train1_file} --dev_file ${dev1_file}

train-S00243-train2-win1-blogDeep5:
	rm -rf models/S00243-train2-win1-blogDeep5
	python ./python/CTP_main.py train S00243-train2-win1-blogDeep5 --model blogDeep --n_epochs 500 --lr=0.001 --lr_decay exp --decay_gamma 0.995 --device cuda:0 --train_file ${train1_file} --dev_file ${dev1_file}

train-S00243-train2-win1-blogDeep6:
	rm -rf models/S00243-train2-win1-blogDeep6
	python ./python/CTP_main.py train S00243-train2-win1-blogDeep6 --model blogDeep --n_epochs 500 --lr=0.01 --lr_decay exp --decay_gamma 0.995 --device cuda:0 --train_file ${train1_file} --dev_file ${dev1_file}

train-S00243-train2-win1-blogDeep7:
	rm -rf models/S00243-train2-win1-blogDeep7
	python ./python/CTP_main.py train S00243-train2-win1-blogDeep7 --model blogDeep --n_epochs 500 --lr=0.005 --lr_decay exp --decay_gamma 0.995 --device cuda:0 --train_file ${train1_file} --dev_file ${dev1_file}

makeDeep:
	make train-S00243-train2-win1-blogDeep1 -B
	make train-S00243-train2-win1-blogDeep2 -B
	make train-S00243-train2-win1-blogDeep3 -B
	make train-S00243-train2-win1-blogDeep4 -B
	make train-S00243-train2-win1-blogDeep5 -B
	make train-S00243-train2-win1-blogDeep6 -B
	make train-S00243-train2-win1-blogDeep7 -B

################################################################################
#################################### Test ######################################
################################################################################
test-S00243-train2-win1-base%:
	python ./python/CTP_main.py test S00243-train2-win1-base$* --model baseline --device cuda:0 --train_file ${train1_file} --dev_file ${dev1_file} --test_file ${test1_file} --baseline_n_hidden 100

test-S00243-train2-win1-fcDeep%:
	python ./python/CTP_main.py test S00243-train2-win1-fcDeep$* --model fc6 --device cuda:0 --train_file ${train1_file} --dev_file ${dev1_file} --test_file ${test1_file}

test-S00243-train2-win1-kir%:
	python ./python/CTP_main.py test S00243-train2-win1-kir$* --model kiranyaz --device cuda:0 --train_file ${train1_file} --dev_file ${dev1_file} --test_file ${test1_file}

test-S00243-train2-win1-blogMed%:
	python ./python/CTP_main.py test S00243-train2-win1-blogMed$* --model blogMed --device cuda:0 --train_file ${train1_file} --dev_file ${dev1_file} --test_file ${test1_file}

test-S00243-train2-win1-blogDeep%:
	python ./python/CTP_main.py test S00243-train2-win1-blogDeep$* --model blogDeep --device cuda:0 --train_file ${train1_file} --dev_file ${dev1_file} --test_file ${test1_file}

makeTest1:
	make test-S00243-train2-win1-base10 -B
	make test-S00243-train2-win1-fcDeep8 -B
	make test-S00243-train2-win1-kir1 -B
	make test-S00243-train2-win1-blogMed1 -B
	make test-S00243-train2-win1-blogDeep1 -B

################################################################################
############################ Prediction on train data ##########################
################################################################################
pred1_file=dat/S00243-train2-win2.h5

predictTrain-S00243-train2-win1-base%:
	python ./python/CTP_main.py predict S00243-train2-win1-base$* --model baseline --patient_file ${pred1_file} --patient_id S00243 --baseline_n_hidden 100

predictTrain-S00243-train2-win1-fcDeep%:
	python ./python/CTP_main.py predict S00243-train2-win1-fcDeep$* --model fc6 --device cuda:0 --patient_file ${pred1_file} --patient_id S00243

predictTrain-S00243-train2-win1-kir%:
	python ./python/CTP_main.py predict S00243-train2-win1-kir$* --model kiranyaz --device cuda:0 --patient_file ${pred1_file} --patient_id S00243

predictTrain-S00243-train2-win1-blogMed%:
	python ./python/CTP_main.py predict S00243-train2-win1-blogMed$* --model blogMed --device cuda:0 --patient_file ${pred1_file} --patient_id S00243

predictTrain-S00243-train2-win1-blogDeep%:
	python ./python/CTP_main.py predict S00243-train2-win1-blogDeep$* --model blogDeep --device cuda:0 --patient_file ${pred1_file} --patient_id S00243

makePredTrain2:
	# make predictTrain-S00243-train2-win1-base1 -B
	# make predictTrain-S00243-train2-win1-base2 -B
	# make predictTrain-S00243-train2-win1-base3 -B
	# make predictTrain-S00243-train2-win1-fcDeep1 -B
	# make predictTrain-S00243-train2-win1-fcDeep4 -B
	# make predictTrain-S00243-train2-win1-fcDeep5 -B
	# make predictTrain-S00243-train2-win1-kir1 -B
	# make predictTrain-S00243-train2-win1-kir2 -B
	# make predictTrain-S00243-train2-win1-kir5 -B
	# make predictTrain-S00243-train2-win1-kir6 -B
	# make predictTrain-S00243-train2-win1-blogMed1 -B
	# make predictTrain-S00243-train2-win1-blogMed3 -B
	# make predictTrain-S00243-train2-win1-blogMed4 -B
	make predictTrain-S00243-train2-win1-blogMed5 -B
	# make predictTrain-S00243-train2-win1-blogMed6 -B

################################################################################
################################# Display ######################################
################################################################################
dispResult2:
	# Training data
	Cp ${pred1_file}_tmax_m.H t1.H
	# Cp models/S00243-train2-win1-base1/S00243-train2-win1-base1_S00243_pred_tmax_sep.H t2a.H
	# Cp models/S00243-train2-win1-base2/S00243-train2-win1-base2_S00243_pred_tmax_sep.H t2b.H
	# Cp models/S00243-train2-win1-base3/S00243-train2-win1-base3_S00243_pred_tmax_sep.H t2c.H
	# Cp models/S00243-train2-win1-fcDeep1/S00243-train2-win1-fcDeep1_S00243_pred_tmax_sep.H t3a.H
	# Cp models/S00243-train2-win1-fcDeep4/S00243-train2-win1-fcDeep4_S00243_pred_tmax_sep.H t3b.H
	# Cp models/S00243-train2-win1-fcDeep5/S00243-train2-win1-fcDeep5_S00243_pred_tmax_sep.H t3c.H
	# Cp models/S00243-train2-win1-kir2/S00243-train2-win1-kir2_S00243_pred_tmax_sep.H t4a.H
	# Cp models/S00243-train2-win1-kir5/S00243-train2-win1-kir5_S00243_pred_tmax_sep.H t4b.H
	# Cp models/S00243-train2-win1-kir6/S00243-train2-win1-kir6_S00243_pred_tmax_sep.H t4c.H
	# Cp models/S00243-train2-win1-kir1/S00243-train2-win1-kir1_S00243_pred_tmax_sep.H t4d.H
	Cp models/S00243-train2-win1-blogMed1/S00243-train2-win1-blogMed1_S00243_pred_tmax_sep.H t5a.H
	Cp models/S00243-train2-win1-blogMed3/S00243-train2-win1-blogMed3_S00243_pred_tmax_sep.H t5b.H
	Cp models/S00243-train2-win1-blogMed4/S00243-train2-win1-blogMed4_S00243_pred_tmax_sep.H t5c.H
	Cp models/S00243-train2-win1-blogMed6/S00243-train2-win1-blogMed6_S00243_pred_tmax_sep.H t5d.H
	Cp models/S00243-train2-win1-blogMed5/S00243-train2-win1-blogMed5_S00243_pred_tmax_sep.H t5e.H
	Cat axis=3 t1.H t2a.H t1.H t2b.H t1.H t2c.H | Grey color=j newclip=1 bclip=-10 eclip=160 grid=y titles="True:Base1:True:Base2:True:Base3" gainpanel=a | Xtpen pixmaps=y &
	Cat axis=3 t1.H t3a.H t1.H t3b.H t1.H t3c.H | Grey color=j newclip=1 bclip=-10 eclip=160 grid=y titles="True:Fc1:True:Fc4:True:Fc5" gainpanel=a | Xtpen pixmaps=y &
	Cat axis=3 t1.H t4a.H t1.H t4b.H t1.H t4c.H t1.H t4d.H | Grey color=j newclip=1 bclip=-10 eclip=160 grid=y titles="True:Kir2:True:Kir5:True:Kir6:True:Kir1" gainpanel=a | Xtpen pixmaps=y &
	Cat axis=3 t1.H t5a.H t1.H t5b.H t1.H t5c.H t1.H t5d.H t1.H t5e.H | Grey color=j newclip=1 bclip=-10 eclip=160 grid=y titles="True:Med1:True:Med3:True:Med4:True:Med6:True:Med5" gainpanel=a | Xtpen pixmaps=y &

dispResult2DiffZones:
	Cp models/S00243-train2-win1-base10/S00243-train2-win1-base10_S00243_pred_tmax_diff_zone.H t2.H
	Cp models/S00243-train2-win1-fcDeep1/S00243-train2-win1-fcDeep1_S00243_pred_tmax_diff_zone.H t3a.H
	Cp models/S00243-train2-win1-fcDeep4/S00243-train2-win1-fcDeep4_S00243_pred_tmax_diff_zone.H t3b.H
	Cp models/S00243-train2-win1-fcDeep4/S00243-train2-win1-fcDeep5_S00243_pred_tmax_diff_zone.H t3c.H
	Cp models/S00243-train2-win1-kir2/S00243-train2-win1-kir2_S00243_pred_tmax_diff_zone.H t4a.H
	Cp models/S00243-train2-win1-kir5/S00243-train2-win1-kir5_S00243_pred_tmax_diff_zone.H t4b.H
	Cp models/S00243-train2-win1-kir6/S00243-train2-win1-kir6_S00243_pred_tmax_diff_zone.H t4c.H
	Cp models/S00243-train2-win1-blogMed1/S00243-train2-win1-blogMed1_S00243_pred_tmax_diff_zone.H t5a.H
	Cp models/S00243-train2-win1-blogMed3/S00243-train2-win1-blogMed3_S00243_pred_tmax_diff_zone.H t5b.H
	Cp models/S00243-train2-win1-blogMed4/S00243-train2-win1-blogMed4_S00243_pred_tmax_diff_zone.H t5c.H
	Cp models/S00243-train2-win1-blogMed6/S00243-train2-win1-blogMed6_S00243_pred_tmax_diff_zone.H t5d.H
	Cat axis=3 t2.H t3a.H t3b.H | Grey color=j newclip=1 bclip=0 eclip=4 grid=y titles="Base:Fc1:Fc4" gainpanel=a wantscalebar=1 | Xtpen pixmaps=y &
	Cat axis=3 t4a.H t4b.H t4c.H | Grey color=j newclip=1 bclip=0 eclip=4 grid=y titles="Kir2:Kir5:Kir6" gainpanel=a wantscalebar=1 | Xtpen pixmaps=y &
	Cat axis=3 t5a.H t5b.H t5c.H t5d.H | Grey color=j newclip=1 bclip=0 eclip=4 grid=y titles="Med1:Med3:Med4:Med6" gainpanel=a wantscalebar=1 | Xtpen pixmaps=y &

################################################################################
############################ Prediction on dev data ############################
################################################################################
pred2_file=dat/S00243-train2-win3.h5

predictDev-S00243-train2-win1-base%:
	python ./python/CTP_main.py predict S00243-train2-win1-base$* --model baseline --patient_file ${pred2_file} --patient_id S00243d --baseline_n_hidden 100

predictDev-S00243-train2-win1-fcDeep%:
	python ./python/CTP_main.py predict S00243-train2-win1-fcDeep$* --model fc6 --device cuda:0 --patient_file ${pred2_file} --patient_id S00243d

predictDev-S00243-train2-win1-kir%:
	python ./python/CTP_main.py predict S00243-train2-win1-kir$* --model kiranyaz --device cuda:0 --patient_file ${pred2_file} --patient_id S00243d

predictDev-S00243-train2-win1-blogMed%:
	python ./python/CTP_main.py predict S00243-train2-win1-blogMed$* --model blogMed --device cuda:0 --patient_file ${pred2_file} --patient_id S00243d

predictDev-S00243-train2-win1-blogDeep%:
	python ./python/CTP_main.py predict S00243-train2-win1-blogDeep$* --model blogDeep --device cuda:0 --patient_file ${pred2_file} --patient_id S00243d

makePredDev2:
	# make predictDev-S00243-train2-win1-base1 -B
	# make predictDev-S00243-train2-win1-base2 -B
	# make predictDev-S00243-train2-win1-base3 -B
	# make predictDev-S00243-train2-win1-fcDeep1 -B
	# make predictDev-S00243-train2-win1-fcDeep4 -B
	# make predictDev-S00243-train2-win1-fcDeep5 -B
	# make predictDev-S00243-train2-win1-kir1 -B
	# make predictDev-S00243-train2-win1-kir2 -B
	# make predictDev-S00243-train2-win1-kir5 -B
	# make predictDev-S00243-train2-win1-kir6 -B
	# make predictDev-S00243-train2-win1-blogMed1 -B
	# make predictDev-S00243-train2-win1-blogMed3 -B
	# make predictDev-S00243-train2-win1-blogMed4 -B
	# make predictDev-S00243-train2-win1-blogMed6 -B
	make predictDev-S00243-train2-win1-blogMed5 -B

dispResult2Dev:
	# Training data
	Cp ${pred2_file}_tmax_m.H t1.H
	# Cp models/S00243-train2-win1-base1/S00243-train2-win1-base1_S00243d_pred_tmax_sep.H t2a.H
	# Cp models/S00243-train2-win1-base2/S00243-train2-win1-base2_S00243d_pred_tmax_sep.H t2b.H
	# Cp models/S00243-train2-win1-base3/S00243-train2-win1-base3_S00243d_pred_tmax_sep.H t2c.H
	# Cp models/S00243-train2-win1-fcDeep1/S00243-train2-win1-fcDeep1_S00243d_pred_tmax_sep.H t3a.H
	# Cp models/S00243-train2-win1-fcDeep4/S00243-train2-win1-fcDeep4_S00243d_pred_tmax_sep.H t3b.H
	# Cp models/S00243-train2-win1-fcDeep5/S00243-train2-win1-fcDeep5_S00243d_pred_tmax_sep.H t3c.H
	# Cp models/S00243-train2-win1-kir2/S00243-train2-win1-kir2_S00243d_pred_tmax_sep.H t4a.H
	# Cp models/S00243-train2-win1-kir5/S00243-train2-win1-kir5_S00243d_pred_tmax_sep.H t4b.H
	# Cp models/S00243-train2-win1-kir6/S00243-train2-win1-kir6_S00243d_pred_tmax_sep.H t4c.H
	# Cp models/S00243-train2-win1-kir1/S00243-train2-win1-kir1_S00243d_pred_tmax_sep.H t4d.H
	Cp models/S00243-train2-win1-blogMed1/S00243-train2-win1-blogMed1_S00243d_pred_tmax_sep.H t5a.H
	Cp models/S00243-train2-win1-blogMed3/S00243-train2-win1-blogMed3_S00243d_pred_tmax_sep.H t5b.H
	Cp models/S00243-train2-win1-blogMed4/S00243-train2-win1-blogMed4_S00243d_pred_tmax_sep.H t5c.H
	Cp models/S00243-train2-win1-blogMed6/S00243-train2-win1-blogMed6_S00243d_pred_tmax_sep.H t5d.H
	Cp models/S00243-train2-win1-blogMed5/S00243-train2-win1-blogMed5_S00243d_pred_tmax_sep.H t5e.H
	# Cat axis=3 t1.H t2a.H t1.H t2b.H t1.H t2c.H | Grey color=j newclip=1 bclip=-10 eclip=160 grid=y titles="True:Base1:True:Base2:True:Base3" gainpanel=a | Xtpen pixmaps=y &
	# Cat axis=3 t1.H t3a.H t1.H t3b.H t1.H t3c.H | Grey color=j newclip=1 bclip=-10 eclip=160 grid=y titles="True:Fc1:True:Fc4:True:Fc5" gainpanel=a | Xtpen pixmaps=y &
	# Cat axis=3 t1.H t4a.H t1.H t4b.H t1.H t4c.H t1.H t4d.H | Grey color=j newclip=1 bclip=-10 eclip=160 grid=y titles="True:Kir2:True:Kir5:True:Kir6:True:Kir1" gainpanel=a | Xtpen pixmaps=y &
	Cat axis=3 t1.H t5a.H t1.H t5b.H t1.H t5c.H t1.H t5d.H t1.H t5e.H | Grey color=j newclip=1 bclip=-10 eclip=160 grid=y titles="True:Med1:True:Med3:True:Med4:True:Med6:True:Med5" gainpanel=a | Xtpen pixmaps=y &

dispResult2DiffZonesDev:
	Cp models/S00243-train2-win1-base10/S00243-train2-win1-base10_S00243d_pred_tmax_diff_zone.H t2.H
	Cp models/S00243-train2-win1-fcDeep1/S00243-train2-win1-fcDeep1_S00243d_pred_tmax_diff_zone.H t3a.H
	Cp models/S00243-train2-win1-fcDeep4/S00243-train2-win1-fcDeep4_S00243d_pred_tmax_diff_zone.H t3b.H
	Cp models/S00243-train2-win1-kir2/S00243-train2-win1-kir2_S00243d_pred_tmax_diff_zone.H t4a.H
	Cp models/S00243-train2-win1-kir5/S00243-train2-win1-kir5_S00243d_pred_tmax_diff_zone.H t4b.H
	Cp models/S00243-train2-win1-kir6/S00243-train2-win1-kir6_S00243d_pred_tmax_diff_zone.H t4c.H
	Cp models/S00243-train2-win1-blogMed1/S00243-train2-win1-blogMed1_S00243d_pred_tmax_diff_zone.H t5a.H
	Cp models/S00243-train2-win1-blogMed3/S00243-train2-win1-blogMed3_S00243d_pred_tmax_diff_zone.H t5b.H
	Cp models/S00243-train2-win1-blogMed4/S00243-train2-win1-blogMed4_S00243d_pred_tmax_diff_zone.H t5c.H
	Cp models/S00243-train2-win1-blogMed6/S00243-train2-win1-blogMed6_S00243d_pred_tmax_diff_zone.H t5d.H
	Cat axis=3 t2.H t3a.H t3b.H | Grey color=j newclip=1 bclip=0 eclip=4 grid=y titles="Base:Fc1:Fc4" gainpanel=a wantscalebar=1 | Xtpen pixmaps=y &
	Cat axis=3 t4a.H t4b.H t4c.H | Grey color=j newclip=1 bclip=0 eclip=4 grid=y titles="Kir2:Kir5:Kir6" gainpanel=a wantscalebar=1 | Xtpen pixmaps=y &
	Cat axis=3 t5a.H t5b.H t5c.H t5d.H | Grey color=j newclip=1 bclip=0 eclip=4 grid=y titles="Med1:Med3:Med4:Med6" gainpanel=a wantscalebar=1 | Xtpen pixmaps=y &
