################################################################################
################################# Training subset 1 ############################
################################################################################
# Training on one subslice

# Train: /net/server2/sep/gbarnier/projects/CTperf/dat/S00243-train-win1.h5
# Dev: /net/server2/sep/gbarnier/projects/CTperf/dat/S00243-train-win1.h5
# Test: /net/server2/sep/gbarnier/projects/CTperf/dat/S00243-train-win2.h5

n_epoch1=200

################################# Baseline #####################################
train-S00243-train-win1-test1:
	rm -rf models/S00243-win1-test1
	python ./python/CTP_main.py train S00243-win1-test1 --model baseline --n_epochs ${n_epoch1} --lr=0.1 --lr_decay exp --device cuda --baseline_n_hidden 100 --train_file dat/S00243-train-win1.h5 --dev_file dat/S00243-train-win2.h5 --device cuda:0

train-S00243-train-win1-test2:
	rm -rf models/S00243-win1-test2
	python ./python/CTP_main.py train S00243-win1-test2 --model baseline --n_epochs ${n_epoch1} --lr=0.1 --lr_decay exp --device cuda --baseline_n_hidden 100 --train_file_list train_file.txt --dev_file_list dev_file.txt --device cuda:0

################################# Baseline #####################################
train-S00243-train-win1-base1:
	rm -rf models/S00243-win1-base1
	python ./python/CTP_main.py train S00243-win1-base1 --model baseline --n_epochs ${n_epoch1} --lr=0.1 --lr_decay exp --device cuda --baseline_n_hidden 100 --train_file1 dat/S00243-train-win1.h5 --dev_file1 dat/S00243-train-win2.h5 --device cuda:0

################################# FC deep ######################################
train-S00243-train-win1-fcDeep1:
	rm -rf models/S00243-win1-fcDeep1
	python ./python/CTP_main.py train S00243-win1-fcDeep1 --model fc6 --n_epochs ${n_epoch1} --lr=0.001 --lr_decay decay --fc6_n_hidden 500 500 500 300 300 --train_file1 dat/S00243-train-win1.h5 --dev_file1 dat/S00243-train-win2.h5 --device cuda:0

# No learning rate decay
train-S00243-train-win1-fcDeep2:
	rm -rf models/S00243-win1-fcDeep2
	python ./python/CTP_main.py train S00243-win1-fcDeep2 --model fc6 --n_epochs ${n_epoch1} --lr=0.001 --fc6_n_hidden 500 500 500 300 300 --train_file1 dat/S00243-train-win1.h5 --dev_file1 dat/S00243-train-win2.h5 --device cuda:0

################################### Blog #######################################
train-S00243-train-win1-blog1:
	rm -rf models/S00243-win1-blog1
	python ./python/CTP_main.py train S00243-win1-blog1 --model blog --n_epochs ${n_epoch1} --lr=0.01 --train_file1 dat/S00243-train-win1.h5 --dev_file1 dat/S00243-train-win2.h5 --device cuda:0

################################# Blog deep ####################################
train-S00243-train-win1-blogDeep1:
	rm -rf models/S00243-win1-blogDeep1
	python ./python/CTP_main.py train S00243-win1-blogDeep1 --model blogDeep --n_epochs ${n_epoch1} --lr=0.01 --train_file1 dat/S00243-train-win1.h5 --dev_file1 dat/S00243-train-win2.h5 --device cuda:0 --lr_decay decay

train-S00243-train-win1-blogDeep2:
	rm -rf models/S00243-win1-blogDeep2
	python ./python/CTP_main.py train S00243-win1-blogDeep2 --model blogDeep --n_epochs ${n_epoch1} --lr=0.005 --train_file1 dat/S00243-train-win1.h5 --dev_file1 dat/S00243-train-win2.h5 --device cuda:1 --batch_size=1024 --lr_decay decay

train-S00243-train-win1-blogDeep3:
	rm -rf models/S00243-win1-blogDeep3
	python ./python/CTP_main.py train S00243-win1-blogDeep3 --model blogDeep --n_epochs ${n_epoch1} --lr=0.005 --train_file1 dat/S00243-train-win1.h5 --dev_file1 dat/S00243-train-win2.h5 --device cuda:1 --batch_size=1024 --lr_decay exp

################################# Blog medium ##################################
train-S00243-train-win1-blogMed1:
	rm -rf models/S00243-win1-blogMed1
	python ./python/CTP_main.py train S00243-win1-blogMed1 --model blogMed --n_epochs ${n_epoch1} --lr=0.01 --train_file1 dat/S00243-train-win1.h5 --dev_file1 dat/S00243-train-win2.h5 --device cuda:0 --lr_decay decay

train-S00243-train-win1-blogMed2:
	rm -rf models/S00243-win1-blogMed2
	python ./python/CTP_main.py train S00243-win1-blogMed2 --model blogMed --n_epochs ${n_epoch1} --lr=0.005 --train_file1 dat/S00243-train-win1.h5 --dev_file1 dat/S00243-train-win2.h5 --device cuda:1 --batch_size=1024 --lr_decay decay

train-S00243-train-win1-blogMed3:
	rm -rf models/S00243-win1-blogMed3
	python ./python/CTP_main.py train S00243-win1-blogMed3 --model blogMed --n_epochs ${n_epoch1} --lr=0.005 --train_file1 dat/S00243-train-win1.h5 --dev_file1 dat/S00243-train-win2.h5 --device cuda:1 --batch_size=1024 --lr_decay exp

train-S00243-train-win1-blogMed4:
	rm -rf models/S00243-win1-blogMed4
	python ./python/CTP_main.py train S00243-win1-blogMed4 --model blogMed --n_epochs ${n_epoch1} --lr=0.001 --train_file1 dat/S00243-train-win1.h5 --dev_file1 dat/S00243-train-win2.h5 --device cuda:1 --batch_size=1024 --lr_decay exp

train-S00243-train-win1-blogMed5:
	rm -rf models/S00243-win1-blogMed5
	python ./python/CTP_main.py train S00243-win1-blogMed5 --model blogMed --n_epochs ${n_epoch1} --lr=0.005 --train_file1 dat/S00243-train-win1.h5 --dev_file1 dat/S00243-train-win2.h5 --device cuda:1 --batch_size=1024 --lr_decay exp

train-S00243-train-win1-blogMed6:
	rm -rf models/S00243-win1-blogMed6
	python ./python/CTP_main.py train S00243-win1-blogMed6 --model blogMed --n_epochs ${n_epoch1} --lr=0.01 --train_file1 dat/S00243-train-win1.h5 --dev_file1 dat/S00243-train-win2.h5 --device cuda:1 --batch_size=1024 --lr_decay exp

train-S00243-train-win1-blogMed7:
	rm -rf models/S00243-win1-blogMed7
	python ./python/CTP_main.py train S00243-win1-blogMed7 --model blogMed --n_epochs ${n_epoch1} --lr=0.0008 --train_file1 dat/S00243-train-win1.h5 --dev_file1 dat/S00243-train-win2.h5 --device cuda:1 --batch_size=1024 --lr_decay decay

#################################### Kir #######################################
train-S00243-train-win1-kir1:
	rm -rf models/S00243-win1-kir1
	python ./python/CTP_main.py train S00243-win1-kir1 --model kiranyaz --n_epochs ${n_epoch1} --lr=0.001 --train_file1 dat/S00243-train-win1.h5 --dev_file1 dat/S00243-train-win2.h5 --device cuda:0 --lr_decay exp

train-S00243-train-win1-kir2:
	rm -rf models/S00243-win1-kir2
	python ./python/CTP_main.py train S00243-win1-kir2 --model kiranyaz --n_epochs ${n_epoch1} --lr=0.001 --lr_decay decay --train_file1 dat/S00243-train-win1.h5 --dev_file1 dat/S00243-train-win2.h5 --device cuda:0

#################################### Gui #######################################
train-S00243-train-win1-gui1:
	rm -rf models/S00243-win1-gui1
	python ./python/CTP_main.py train S00243-win1-gui1 --model gui --n_epochs ${n_epoch1} --lr=0.01  --lr_decay decay --train_file1 dat/S00243-train-win1.h5 --dev_file1 dat/S00243-train-win2.h5 --device cuda:0

# Learning rate decay
train-S00243-train-win1-gui2:
	rm -rf models/S00243-win1-gui2
	python ./python/CTP_main.py train S00243-win1-gui2 --model gui --n_epochs ${n_epoch1} --lr=0.01 --train_file1 dat/S00243-train-win1.h5 --dev_file1 dat/S00243-train-win2.h5 --device cuda:0

############################## Launch training #################################
makeTrainWin1:
	make train-S00243-train-win1-base1 -B
	make train-S00243-train-win1-fcDeep1 -B
	make train-S00243-train-win1-fcDeep2 -B
	make train-S00243-train-win1-blog1 -B
	make train-S00243-train-win1-blogDeep1 -B
	make train-S00243-train-win1-kir1 -B
	make train-S00243-train-win1-kir2 -B
	make train-S00243-train-win1-gui1 -B
	make train-S00243-train-win1-gui2 -B

################################################################################
################################### Test #######################################
################################################################################
test-S00243-train-win1-config1:
	python ./python/CTP_main.py test S00243-win1-config1 --model baseline

################################# Baseline #####################################
test-S00243-train-win1-test1:
	python ./python/CTP_main.py test S00243-win1-test1 --model baseline --device cuda --baseline_n_hidden 100 --train_file dat/S00243-train-win1.h5 --dev_file dat/S00243-train-win2.h5 --test_file dat/S00243-train-win4.h5 --device cuda:0

test-S00243-train-win1-test2:
	python ./python/CTP_main.py test S00243-win1-test2 --model baseline --n_epochs ${n_epoch1} --lr=0.1 --lr_decay exp --device cuda --baseline_n_hidden 100 --train_file_list train_file.txt --dev_file_list dev_file.txt --test_file_list test_file.txt --device cuda:0 

################################################################################
################################# Prediction ###################################
################################################################################
predict-S00243-train-win1-base1:
	python ./python/CTP_main.py predict S00243-win1-base1 --model baseline --patient_file dat/S00243-train-win2.h5 --patient_id S00243

predict-S00243-train-win1-fcDeep1:
	python ./python/CTP_main.py predict S00243-win1-fcDeep1 --model fc6 --patient_file dat/S00243-train-win2.h5 --patient_id S00243 --fc6_n_hidden 500 500 500 300 300

predict-S00243-train-win1-fcDeep2:
	python ./python/CTP_main.py predict S00243-win1-fcDeep2 --model fc6 --patient_file dat/S00243-train-win2.h5 --patient_id S00243 --fc6_n_hidden 500 500 500 300 300

predict-S00243-train-win1-blog1:
	python ./python/CTP_main.py predict S00243-win1-blog1 --model blog --patient_file dat/S00243-train-win2.h5 --patient_id S00243

predict-S00243-train-win1-blog2:
	python ./python/CTP_main.py predict S00243-win1-blog2 --model blog --patient_file dat/S00243-train-win2.h5 --patient_id S00243

predict-S00243-train-win1-kir1:
	python ./python/CTP_main.py predict S00243-win1-kir1 --model kiranyaz --patient_file dat/S00243-train-win2.h5 --patient_id S00243

predict-S00243-train-win1-kir2:
	python ./python/CTP_main.py predict S00243-win1-kir2 --model kiranyaz --patient_file dat/S00243-train-win2.h5 --patient_id S00243

predict-S00243-train-win1-gui1:
	python ./python/CTP_main.py predict S00243-win1-gui1 --model gui --patient_file dat/S00243-train-win2.h5 --patient_id S00243

predict-S00243-train-win1-gui2:
	python ./python/CTP_main.py predict S00243-win1-gui2 --model gui --patient_file dat/S00243-train-win2.h5 --patient_id S00243

makePredWin1:
	make predict-S00243-train-win1-base1 -B
	make predict-S00243-train-win1-fcDeep1 -B
	make predict-S00243-train-win1-fcDeep2 -B
	make predict-S00243-train-win1-blog1 -B
	make predict-S00243-train-win1-blog2 -B
	make predict-S00243-train-win1-kir1 -B
	make predict-S00243-train-win1-kir2 -B
	make predict-S00243-train-win1-gui1 -B
	make predict-S00243-train-win1-gui2 -B

################################################################################
################################# Display ######################################
################################################################################
dispResult:
	# True
	Cp dat/S00243-train-win1.h5_tmax_m.H t0.H
	# Baseline
	Cp models/S00243-win1-config1/S00243-win1-config1_S00243_pred_tmax_sep.H t1.H
	# FC 6
	Cp models/S00243-win1-config2/S00243-win1-config2_S00243_pred_tmax_sep.H t2.H
	Cp models/S00243-win1-config3/S00243-win1-config3_S00243_pred_tmax_sep.H t3.H
	# Blog
	Cp models/S00243-win1-config4/S00243-win1-config4_S00243_pred_tmax_sep.H t4.H
	Cp models/S00243-win1-config5/S00243-win1-config5_S00243_pred_tmax_sep.H t5.H
	# Kir
	Cp models/S00243-win1-config6/S00243-win1-config6_S00243_pred_tmax_sep.H t6.H
	Cp models/S00243-win1-config6b/S00243-win1-config6b_S00243_pred_tmax_sep.H t6b.H
	# Gui
	Cp models/S00243-win1-config7/S00243-win1-config7_S00243_pred_tmax_sep.H t7.H
	Cp models/S00243-win1-config8/S00243-win1-config8_S00243_pred_tmax_sep.H t8.H
	# Display
	Cat axis=3 t0.H t1.H t0.H t2.H t0.H t3.H t0.H t4.H t0.H t5.H t0.H t6.H t0.H t6b.H t0.H t7.H t0.H t8.H | Grey bclip=0 eclip=450 grid=y gainpanel=a color=j newclip=1 titles="True:Base:True:Fc6:True:Fc6-decay:True:Blog:True:Blog-decay:True:Kir:True:Kir-decay:True:Gui-decay:True:Gui" wantscalebar=1 | Xtpen pixmaps=y &
	# 1D profiles
	# Window3d n2=1 min2=100 < t1.H  > p1.H
	# Window3d n2=1 min2=100 < t2.H  > p2.H
	# Window3d n2=1 min2=100 < t3.H  > p3.H
	# Window3d n2=1 min2=100 < t4.H  > p4.H
	# Window3d n2=1 min2=100 < t5.H  > p5.H
	# Window3d n2=1 min2=100 < t6.H  > p6.H
	# Window3d n2=1 min2=100 < t7.H > p7.H
	# Window3d n2=1 min2=100 < t8.H > p8.H
	# Cat axis=2 p1.H p2.H p3.H | Graph grid=y legend=y curvelabel="True:FCRD:FC" | Xtpen &
	# Cat axis=2 p1.H p4.H p5.H | Graph grid=y legend=y curvelabel="True:BLOG:BLOGRD" | Xtpen &
	# Cat axis=2 p1.H p6.H | Graph grid=y legend=y curvelabel="True:KIR" | Xtpen &
	# Cat axis=2 p1.H p7.H p8.H | Graph grid=y legend=y curvelabel="True:GUIRD:GUI"| Xtpen &
