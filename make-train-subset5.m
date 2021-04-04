################################################################################
################################# Training subset 5 ############################
################################################################################
# Training/dev/test on one full slice
train5_file=par/train_file5.txt
dev5_file=par/dev_file5.txt
test5_file=par/test_file5.txt
n_epoch5=350

################################## Baseline ####################################
train5-win1-base1:
	rm -rf models/train5-win1-base1
	python ./python/CTP_main.py train train5-win1-base1 --model baseline --n_epochs 100 --lr=1e-6 --device cuda:0 --train_file_list ${train5_file} --dev_file_list ${dev5_file} --baseline_n_hidden 100

train5-win1-base2:
	rm -rf models/train5-win1-base2
	python ./python/CTP_main.py train train5-win1-base2 --model baseline --n_epochs 100 --lr=1e-5 --device cuda:1 --train_file_list ${train5_file} --dev_file_list ${dev5_file} --baseline_n_hidden 100

train5-win1-base3:
	rm -rf models/train5-win1-base3
	python ./python/CTP_main.py train train5-win1-base3 --model baseline --n_epochs 100 --lr=1e-4 --device cuda:2 --train_file_list ${train5_file} --dev_file_list ${dev5_file} --baseline_n_hidden 100

train5-win1-base4:
	rm -rf models/train5-win1-base4
	python ./python/CTP_main.py train train5-win1-base4 --model baseline --n_epochs 100 --lr=1e-3 --device cuda:3 --train_file_list ${train5_file} --dev_file_list ${dev5_file} --baseline_n_hidden 100

train5-win1-base5:
	rm -rf models/train5-win1-base5
	python ./python/CTP_main.py train train5-win1-base5 --model baseline --n_epochs 100 --lr=0.01 --device cuda:0 --train_file_list ${train5_file} --dev_file_list ${dev5_file} --baseline_n_hidden 100

train5-win1-base6:
	rm -rf models/train5-win1-base6
	python ./python/CTP_main.py train train5-win1-base6 --model baseline --n_epochs 100 --lr=0.1 --device cuda:1 --train_file_list ${train5_file} --dev_file_list ${dev5_file} --baseline_n_hidden 100

#################################### FC ########################################
train5-win1-fc1:
	rm -rf models/train5-win1-fc1
	python ./python/CTP_main.py train train5-win1-fc1 --model fc6 --n_epochs 50 --lr=0.00001 --device cuda:0 --train_file_list ${train5_file} --dev_file_list ${dev5_file}

train5-win1-fc2:
	rm -rf models/train5-win1-fc2
	python ./python/CTP_main.py train train5-win1-fc2 --model fc6 --n_epochs 50 --lr=0.00005 --device cuda:1 --train_file_list ${train5_file} --dev_file_list ${dev5_file}

train5-win1-fc3:
	rm -rf models/train5-win1-fc3
	python ./python/CTP_main.py train train5-win1-fc3 --model fc6 --n_epochs 50 --lr=0.0001 --device cuda:2 --train_file_list ${train5_file} --dev_file_list ${dev5_file}

train5-win1-fc4:
	rm -rf models/train5-win1-fc4
	python ./python/CTP_main.py train train5-win1-fc4 --model fc6 --n_epochs 50 --lr=0.00025 --device cuda:3 --train_file_list ${train5_file} --dev_file_list ${dev5_file}

train5-win1-fc5:
	rm -rf models/train5-win1-fc5
	python ./python/CTP_main.py train train5-win1-fc5 --model fc6 --n_epochs 50 --lr=0.0005 --device cuda:0 --train_file_list ${train5_file} --dev_file_list ${dev5_file}

train5-win1-fc6:
	rm -rf models/train5-win1-fc6
	python ./python/CTP_main.py train train5-win1-fc6 --model fc6 --n_epochs 50 --lr=0.0008 --device cuda:1 --train_file_list ${train5_file} --dev_file_list ${dev5_file}

train5-win1-fc7:
	rm -rf models/train5-win1-fc7
	python ./python/CTP_main.py train train5-win1-fc7 --model fc6 --n_epochs 50 --lr=0.001 --device cuda:2 --train_file_list ${train5_file} --dev_file_list ${dev5_file}

train5-win1-fc8:
	rm -rf models/train5-win1-fc8
	python ./python/CTP_main.py train train5-win1-fc8 --model fc6 --n_epochs 50 --lr=0.001 --device cuda:3 --train_file_list ${train5_file} --dev_file_list ${dev5_file} --loss huber

train5-win1-fc9:
	rm -rf models/train5-win1-fc9
	python ./python/CTP_main.py train train5-win1-fc9 --model fc6 --n_epochs 50 --lr=1.0e-6 --device cuda:0 --train_file_list ${train5_file} --dev_file_list ${dev5_file}

train5-win1-fc10:
	rm -rf models/train5-win1-fc10
	python ./python/CTP_main.py train train5-win1-fc10 --model fc6 --n_epochs 50 --lr=2.5e-6 --device cuda:1 --train_file_list ${train5_file} --dev_file_list ${dev5_file}

train5-win1-fc11:
	rm -rf models/train5-win1-fc11
	python ./python/CTP_main.py train train5-win1-fc11 --model fc6 --n_epochs 50 --lr=5.0e-6 --device cuda:2 --train_file_list ${train5_file} --dev_file_list ${dev5_file}

train5-win1-fc12:
	rm -rf models/train5-win1-fc12
	python ./python/CTP_main.py train train5-win1-fc12 --model fc6 --n_epochs 50 --lr=7.5e-6 --device cuda:3 --train_file_list ${train5_file} --dev_file_list ${dev5_file}

train5-win1-fc13:
	rm -rf models/train5-win1-fc13
	python ./python/CTP_main.py train train5-win1-fc13 --model fc6 --n_epochs 200 --lr=1.0e-6 --device cuda:0 --train_file_list ${train5_file} --dev_file_list ${dev5_file}

train5-win1-fc14:
	rm -rf models/train5-win1-fc14
	python ./python/CTP_main.py train train5-win1-fc14 --model fc6 --n_epochs 200 --lr=0.75e-6 --device cuda:1 --train_file_list ${train5_file} --dev_file_list ${dev5_file}

train5-win1-fc15:
	rm -rf models/train5-win1-fc15
	python ./python/CTP_main.py train train5-win1-fc15 --model fc6 --n_epochs 200 --lr=0.5e-6 --device cuda:2 --train_file_list ${train5_file} --dev_file_list ${dev5_file}

train5-win1-fc16:
	rm -rf models/train5-win1-fc16
	python ./python/CTP_main.py train train5-win1-fc16 --model fc6 --n_epochs 200 --lr=0.25e-6 --device cuda:3 --train_file_list ${train5_file} --dev_file_list ${dev5_file}

train5-win1-fcHuber1:
	rm -rf models/train5-win1-fcHuber1
	python ./python/CTP_main.py train train5-win1-fcHuber1 --model fc6 --n_epochs 50 --lr=1.0e-6 --device cuda:0 --train_file_list ${train5_file} --dev_file_list ${dev5_file} --loss huber

train5-win1-fcHuber2:
	rm -rf models/train5-win1-fcHuber2
	python ./python/CTP_main.py train train5-win1-fcHuber2 --model fc6 --n_epochs 50 --lr=1.0e-5 --device cuda:1 --train_file_list ${train5_file} --dev_file_list ${dev5_file} --loss huber

train5-win1-fcHuber3:
	rm -rf models/train5-win1-fcHuber3
	python ./python/CTP_main.py train train5-win1-fcHuber3 --model fc6 --n_epochs 50 --lr=1.0e-4 --device cuda:2 --train_file_list ${train5_file} --dev_file_list ${dev5_file} --loss huber

train5-win1-fcHuber4:
	rm -rf models/train5-win1-fcHuber4
	python ./python/CTP_main.py train train5-win1-fcHuber4 --model fc6 --n_epochs 50 --lr=1.0e-3 --device cuda:3 --train_file_list ${train5_file} --dev_file_list ${dev5_file} --loss huber

train5-win1-fcHuber5:
	rm -rf models/train5-win1-fcHuber5
	python ./python/CTP_main.py train train5-win1-fcHuber5 --model fc6 --n_epochs 50 --lr=1.0e-2 --device cuda:0 --train_file_list ${train5_file} --dev_file_list ${dev5_file} --loss huber

train5-win1-fcMae1:
	rm -rf models/train5-win1-fcMae1
	python ./python/CTP_main.py train train5-win1-fcMae1 --model fc6 --n_epochs 50 --lr=1.0e-6 --device cuda:0 --train_file_list ${train5_file} --dev_file_list ${dev5_file} --loss mae

train5-win1-fcMae2:
	rm -rf models/train5-win1-fcMae2
	python ./python/CTP_main.py train train5-win1-fcMae2 --model fc6 --n_epochs 50 --lr=1.0e-5 --device cuda:1 --train_file_list ${train5_file} --dev_file_list ${dev5_file} --loss mae

train5-win1-fcMae3:
	rm -rf models/train5-win1-fcMae3
	python ./python/CTP_main.py train train5-win1-fcMae3 --model fc6 --n_epochs 50 --lr=1.0e-4 --device cuda:2 --train_file_list ${train5_file} --dev_file_list ${dev5_file} --loss mae

train5-win1-fcMae4:
	rm -rf models/train5-win1-fcMae4
	python ./python/CTP_main.py train train5-win1-fcMae4 --model fc6 --n_epochs 50 --lr=1.0e-3 --device cuda:3 --train_file_list ${train5_file} --dev_file_list ${dev5_file} --loss mae

train5-win1-fcMae5:
	rm -rf models/train5-win1-fcMae5
	python ./python/CTP_main.py train train5-win1-fcMae4 --model fc6 --n_epochs 50 --lr=1.0e-2 --device cuda:3 --train_file_list ${train5_file} --dev_file_list ${dev5_file} --loss mae

#################################### Kir #######################################
train5-win1-kir1:
	rm -rf models/train5-win1-kir1
	python ./python/CTP_main.py train train5-win1-kir1 --model kiranyaz --n_epochs 50 --lr=1.0e-6 --device cuda:0 --train_file_list ${train5_file} --dev_file_list ${dev5_file}

train5-win1-kir2:
	rm -rf models/train5-win1-kir2
	python ./python/CTP_main.py train train5-win1-kir2 --model kiranyaz --n_epochs 50 --lr=1.0e-5 --device cuda:1 --train_file_list ${train5_file} --dev_file_list ${dev5_file}

train5-win1-kir3:
	rm -rf models/train5-win1-kir3
	python ./python/CTP_main.py train train5-win1-kir3 --model kiranyaz --n_epochs 50 --lr=1.0e-4 --device cuda:2 --train_file_list ${train5_file} --dev_file_list ${dev5_file}

train5-win1-kir4:
	rm -rf models/train5-win1-kir4
	python ./python/CTP_main.py train train5-win1-kir4 --model kiranyaz --n_epochs 50 --lr=1.0e-3 --device cuda:3 --train_file_list ${train5_file} --dev_file_list ${dev5_file}

train5-win1-kir5:
	rm -rf models/train5-win1-kir5
	python ./python/CTP_main.py train train5-win1-kir5 --model kiranyaz --n_epochs 50 --lr=1.0e-2 --device cuda:0 --train_file_list ${train5_file} --dev_file_list ${dev5_file}

train5-win1-kir6:
	rm -rf models/train5-win1-kir6
	python ./python/CTP_main.py train train5-win1-kir6 --model kiranyaz --n_epochs 50 --lr=0.1 --device cuda:1 --train_file_list ${train5_file} --dev_file_list ${dev5_file}

train5-win1-kir7:
	rm -rf models/train5-win1-kir7
	python ./python/CTP_main.py train train5-win1-kir7 --model kiranyaz --n_epochs 100 --lr=1.0e-4 --device cuda:0 --train_file_list ${train5_file} --dev_file_list ${dev5_file} --lr_decay step --step_size 10 --decay_gamma 0.999

train5-win1-kir8:
	rm -rf models/train5-win1-kir8
	python ./python/CTP_main.py train train5-win1-kir8 --model kiranyaz --n_epochs 100 --lr=1.0e-4 --device cuda:1 --train_file_list ${train5_file} --dev_file_list ${dev5_file} --lr_decay decay --decay_rate 0.001

train5-win1-kir9:
	rm -rf models/train5-win1-kir9
	python ./python/CTP_main.py train train5-win1-kir9 --model kiranyaz --n_epochs 100 --lr=1.0e-4 --device cuda:2 --train_file_list ${train5_file} --dev_file_list ${dev5_file}

train5-win1-kir10:
	rm -rf models/train5-win1-kir10
	python ./python/CTP_main.py train train5-win1-kir10 --model kiranyaz --n_epochs 100 --lr=1.0e-4 --device cuda:3 --train_file_list ${train5_file} --dev_file_list ${dev5_file} --lr_decay step --step_size 10 --decay_gamma 0.99

train5-win1-kir11:
	rm -rf models/train5-win1-kir11
	python ./python/CTP_main.py train train5-win1-kir11 --model kiranyaz --n_epochs 100 --lr=2.0e-4 --device cuda:0 --train_file_list ${train5_file} --dev_file_list ${dev5_file}

train5-win1-kir12:
	rm -rf models/train5-win1-kir12
	python ./python/CTP_main.py train train5-win1-kir12 --model kiranyaz --n_epochs 100 --lr=3.0e-4 --device cuda:1 --train_file_list ${train5_file} --dev_file_list ${dev5_file}

train5-win1-kir13:
	rm -rf models/train5-win1-kir13
	python ./python/CTP_main.py train train5-win1-kir13 --model kiranyaz --n_epochs 100 --lr=4.0e-4 --device cuda:2 --train_file_list ${train5_file} --dev_file_list ${dev5_file}

train5-win1-kir14:
	rm -rf models/train5-win1-kir14
	python ./python/CTP_main.py train train5-win1-kir14 --model kiranyaz --n_epochs 100 --lr=0.5e-4 --device cuda:3 --train_file_list ${train5_file} --dev_file_list ${dev5_file}

train5-win1-kirMae1:
	rm -rf models/train5-win1-kirMae1
	python ./python/CTP_main.py train train5-win1-kirMae1 --model kiranyaz --n_epochs 100 --lr=0.1e-4 --device cuda:0 --train_file_list ${train5_file} --dev_file_list ${dev5_file}	--loss mae

train5-win1-kirMae2:
	rm -rf models/train5-win1-kirMae2
	python ./python/CTP_main.py train train5-win1-kirMae2 --model kiranyaz --n_epochs 100 --lr=1e-4 --device cuda:1 --train_file_list ${train5_file} --dev_file_list ${dev5_file}	--loss mae

train5-win1-kirMae3:
	rm -rf models/train5-win1-kirMae3
	python ./python/CTP_main.py train train5-win1-kirMae3 --model kiranyaz --n_epochs 100 --lr=1.5e-4 --device cuda:2 --train_file_list ${train5_file} --dev_file_list ${dev5_file}	--loss mae

train5-win1-kirMae4:
	rm -rf models/train5-win1-kirMae4
	python ./python/CTP_main.py train train5-win1-kirMae4 --model kiranyaz --n_epochs 100 --lr=0.5e-4 --device cuda:3 --train_file_list ${train5_file} --dev_file_list ${dev5_file}	--loss mae

train5-win1-kirHuber1:
	rm -rf models/train5-win1-kirHuber1
	python ./python/CTP_main.py train train5-win1-kirHuber1 --model kiranyaz --n_epochs 100 --lr=0.1e-4 --device cuda:0 --train_file_list ${train5_file} --dev_file_list ${dev5_file}	--loss huber

train5-win1-kirHuber2:
	rm -rf models/train5-win1-kirHuber2
	python ./python/CTP_main.py train train5-win1-kirHuber2 --model kiranyaz --n_epochs 100 --lr=1e-4 --device cuda:1 --train_file_list ${train5_file} --dev_file_list ${dev5_file}	--loss huber

train5-win1-kirHuber3:
	rm -rf models/train5-win1-kirHuber3
	python ./python/CTP_main.py train train5-win1-kirHuber3 --model kiranyaz --n_epochs 100 --lr=1.5e-4 --device cuda:2 --train_file_list ${train5_file} --dev_file_list ${dev5_file} --loss huber

train5-win1-kirHuber4:
	rm -rf models/train5-win1-kirHuber4
	python ./python/CTP_main.py train train5-win1-kirHuber4 --model kiranyaz --n_epochs 100 --lr=0.5e-4 --device cuda:3 --train_file_list ${train5_file} --dev_file_list ${dev5_file} --loss huber

################################### Blog med ###################################
# MSE + Adam + No lr decay
train5-win1-blog1:
	rm -rf models/train5-win1-blog1
	python ./python/CTP_main.py train train5-win1-blog1 --model blog --n_epochs 100 --lr=1e-6 --device cuda:0 --train_file_list ${train5_file} --dev_file_list ${dev5_file}

train5-win1-blog2:
	rm -rf models/train5-win1-blog2
	python ./python/CTP_main.py train train5-win1-blog2 --model blog --n_epochs 100 --lr=1e-5 --device cuda:1 --train_file_list ${train5_file} --dev_file_list ${dev5_file}

train5-win1-blog3:
	rm -rf models/train5-win1-blog3
	python ./python/CTP_main.py train train5-win1-blog3 --model blog --n_epochs 100 --lr=1e-4 --device cuda:2 --train_file_list ${train5_file} --dev_file_list ${dev5_file}

train5-win1-blog4:
	rm -rf models/train5-win1-blog4
	python ./python/CTP_main.py train train5-win1-blog4 --model blog --n_epochs 100 --lr=5e-6 --device cuda:3 --train_file_list ${train5_file} --dev_file_list ${dev5_file}

train5-win1-blog5:
	rm -rf models/train5-win1-blog5
	python ./python/CTP_main.py train train5-win1-blog5 --model blog --n_epochs 100 --lr=2.5e-5 --device cuda:0 --train_file_list ${train5_file} --dev_file_list ${dev5_file}

train5-win1-blog6:
	rm -rf models/train5-win1-blog6
	python ./python/CTP_main.py train train5-win1-blog6 --model blog --n_epochs 200 --lr=1e-6 --device cuda:1 --train_file_list ${train5_file} --dev_file_list ${dev5_file}

train5-win1-blog7:
	rm -rf models/train5-win1-blog7
	python ./python/CTP_main.py train train5-win1-blog7 --model blog --n_epochs 200 --lr=0.5e-6 --device cuda:2 --train_file_list ${train5_file} --dev_file_list ${dev5_file}

train5-win1-blog8:
	rm -rf models/train5-win1-blog8
	python ./python/CTP_main.py train train5-win1-blog8 --model blog --n_epochs 200 --lr=0.25e-6 --device cuda:3 --train_file_list ${train5_file} --dev_file_list ${dev5_file}

train5-win1-blog9:
	rm -rf models/train5-win1-blog9
	python ./python/CTP_main.py train train5-win1-blog9 --model blog --n_epochs 200 --lr=1e-5 --device cuda:1 --train_file_list ${train5_file} --dev_file_list ${dev5_file}

train5-win1-blog10:
	rm -rf models/train5-win1-blog10
	python ./python/CTP_main.py train train5-win1-blog10 --model blog --n_epochs 40 --lr=1e-4 --device cuda:0 --train_file_list ${train5_file} --dev_file_list ${dev5_file}

train5-win1-blog11:
	rm -rf models/train5-win1-blog11
	python ./python/CTP_main.py train train5-win1-blog11 --model blog --n_epochs 200 --lr=1e-4 --device cuda:2 --train_file_list ${train5_file} --dev_file_list ${dev5_file}

train5-win1-blog12:
	rm -rf models/train5-win1-blog12
	python ./python/CTP_main.py train train5-win1-blog12 --model blog --n_epochs 100 --lr=7.5e-6 --device cuda:3 --train_file_list ${train5_file} --dev_file_list ${dev5_file}

################################################################################
############################ Prediction on train data ##########################
################################################################################
# pred5_file_train=dat/S00293-train5-win1.h5
# pred5_file_train=dat/S000267-train4-win1.h5
# pred5_file_train=dat/S00295-train4-win1.h5
# pred5_file_train=dat/S00275-train4-win1.h5
# pred5_file_train=dat/S00291-train5-win1.h5
# pred5_file_train=dat/S00289-train5-win1.h5
pred5_file_train=dat/S00288-train5-win1.h5

predictTrain5-win1-base%:
	python ./python/CTP_main.py predict train5-win1-base$* --model baseline --patient_file ${pred5_file_train} --patient_id S00243 --baseline_n_hidden 100 --device cuda:0

predictTrain5-win1-fc%:
	python ./python/CTP_main.py predict train5-win1-fc$* --model fc6 --device cuda:0 --patient_file ${pred5_file_train} --patient_id S00243

predictTrain5-win1-kir%:
	python ./python/CTP_main.py predict train5-win1-kir$* --model kiranyaz --device cuda:0 --patient_file ${pred5_file_train} --patient_id S00243

makePredTrain5:
	# make predictTrain5-win1-base1 -B
	# make predictTrain5-win1-fc1 -B
	# make predictTrain5-win1-fc2 -B
	# make predictTrain5-win1-fc3 -B
	# make predictTrain5-win1-fc4 -B
	# make predictTrain5-win1-fc5 -B
	# make predictTrain5-win1-fc6 -B
	# make predictTrain5-win1-fc7 -B
	# make predictTrain5-win1-fc8 -B
	# make predictTrain5-win1-fc9 -B
	# make predictTrain5-win1-fc10 -B
	# make predictTrain5-win1-fc11 -B
	# make predictTrain5-win1-fc12 -B
	# make predictTrain5-win1-fc13 -B
	# make predictTrain5-win1-fc14 -B
	# make predictTrain5-win1-fc15 -B
	# make predictTrain5-win1-fc16 -B
	# make predictTrain5-win1-kir1 -B
	# make predictTrain5-win1-kir2 -B
	# make predictTrain5-win1-kir3 -B
	# make predictTrain5-win1-kir4 -B
	# make predictTrain5-win1-kir7 -B
	# make predictTrain5-win1-kir8 -B
	# make predictTrain5-win1-kir9 -B
	# make predictTrain5-win1-kir10 -B
	# make predictTrain5-win1-kir11 -B
	# make predictTrain5-win1-kir12 -B
	# make predictTrain5-win1-kir13 -B
	# make predictTrain5-win1-kir14 -B

################################################################################
################################# Display ######################################
################################################################################
# type5=_avg
type5=
eclip5=300
bclip5=0
dispResult5-s%:
	# Training data
	Window3d n3=1 f3=$* < ${pred5_file_train}_tmax_m.H > t1.H
	# Base
	Window3d n3=1 f3=$* < models/train5-win1-base1/train5-win1-base1_S00243_pred_tmax${type5}_sep.H > t2.H
	# FC
	Window3d n3=1 f3=$* < models/train5-win1-fc1/train5-win1-fc1_S00243_pred_tmax${type5}_sep.H > t3a.H
	Window3d n3=1 f3=$* < models/train5-win1-fc2/train5-win1-fc2_S00243_pred_tmax${type5}_sep.H > t3b.H
	Window3d n3=1 f3=$* < models/train5-win1-fc3/train5-win1-fc3_S00243_pred_tmax${type5}_sep.H > t3c.H
	Window3d n3=1 f3=$* < models/train5-win1-fc4/train5-win1-fc4_S00243_pred_tmax${type5}_sep.H > t3d.H
	Window3d n3=1 f3=$* < models/train5-win1-fc5/train5-win1-fc5_S00243_pred_tmax${type5}_sep.H > t3e.H
	Window3d n3=1 f3=$* < models/train5-win1-fc6/train5-win1-fc6_S00243_pred_tmax${type5}_sep.H > t3f.H
	Window3d n3=1 f3=$* < models/train5-win1-fc7/train5-win1-fc7_S00243_pred_tmax${type5}_sep.H > t3g.H
	Window3d n3=1 f3=$* < models/train5-win1-fc8/train5-win1-fc8_S00243_pred_tmax${type5}_sep.H > t3h.H
	Window3d n3=1 f3=$* < models/train5-win1-fc9/train5-win1-fc9_S00243_pred_tmax${type5}_sep.H > t3i.H
	Window3d n3=1 f3=$* < models/train5-win1-fc10/train5-win1-fc10_S00243_pred_tmax${type5}_sep.H > t3j.H
	Window3d n3=1 f3=$* < models/train5-win1-fc11/train5-win1-fc11_S00243_pred_tmax${type5}_sep.H > t3k.H
	Window3d n3=1 f3=$* < models/train5-win1-fc12/train5-win1-fc12_S00243_pred_tmax${type5}_sep.H > t3l.H
	Window3d n3=1 f3=$* < models/train5-win1-fc13/train5-win1-fc13_S00243_pred_tmax${type5}_sep.H > t3m.H
	Window3d n3=1 f3=$* < models/train5-win1-fc14/train5-win1-fc14_S00243_pred_tmax${type5}_sep.H > t3n.H
	Window3d n3=1 f3=$* < models/train5-win1-fc15/train5-win1-fc15_S00243_pred_tmax${type5}_sep.H > t3o.H
	Window3d n3=1 f3=$* < models/train5-win1-fc16/train5-win1-fc16_S00243_pred_tmax${type5}_sep.H > t3p.H
	# Kir
	Window3d n3=1 f3=$* < models/train5-win1-kir1/train5-win1-kir1_S00243_pred_tmax${type5}_sep.H > t4a.H
	Window3d n3=1 f3=$* < models/train5-win1-kir2/train5-win1-kir2_S00243_pred_tmax${type5}_sep.H > t4b.H
	Window3d n3=1 f3=$* < models/train5-win1-kir3/train5-win1-kir3_S00243_pred_tmax${type5}_sep.H > t4c.H
	Window3d n3=1 f3=$* < models/train5-win1-kir4/train5-win1-kir4_S00243_pred_tmax${type5}_sep.H > t4d.H
	# Window3d n3=1 f3=$* < models/train5-win1-kir5/train5-win1-kir5_S00243_pred_tmax${type5}_sep.H > t4e.H
	# Window3d n3=1 f3=$* < models/train5-win1-kir6/train5-win1-kir6_S00243_pred_tmax${type5}_sep.H > t4f.H
	Window3d n3=1 f3=$* < models/train5-win1-kir7/train5-win1-kir7_S00243_pred_tmax${type5}_sep.H > t4g.H
	Window3d n3=1 f3=$* < models/train5-win1-kir8/train5-win1-kir8_S00243_pred_tmax${type5}_sep.H > t4h.H
	Window3d n3=1 f3=$* < models/train5-win1-kir9/train5-win1-kir9_S00243_pred_tmax${type5}_sep.H > t4i.H
	Window3d n3=1 f3=$* < models/train5-win1-kir10/train5-win1-kir10_S00243_pred_tmax${type5}_sep.H > t4j.H
	# Display FC
	# Cat axis=3 t1.H t2.H t1.H t3a.H t1.H t3b.H t1.H t3c.H t1.H t3d.H t1.H t3e.H t1.H t3f.H t1.H t3g.H t1.H t3h.H t1.H t3i.H | Grey color=j newclip=1 bclip=${bclip5} eclip=${eclip5} grid=y titles="True:Base:True:Fc1:True:Fc2:True:Fc3:True:Fc4:True:Fc5:True:Fc6:True:Fc7:True:Fc8:True:Fc9" gainpanel=a | Xtpen pixmaps=y &
	# Cat axis=3 t1.H t3k.H t1.H t3l.H t1.H t3m.H t1.H t3n.H t1.H t3o.H t1.H t3p.H | Grey color=j newclip=1 bclip=${bclip5} eclip=${eclip5} grid=y titles="True:Fc10:True:Fc11:True:Fc12:True:Fc13:True:Fc14:True:Fc15:True:Fc16" gainpanel=a | Xtpen pixmaps=y &
	# Cat axis=3 t1.H t4a.H t1.H t4b.H t1.H t4c.H t1.H t4d.H t1.H t4g.H t1.H t4h.H t1.H t4i.H t1.H t4j.H | Grey color=j newclip=1 bclip=${bclip5} eclip=${eclip5} grid=y titles="True:Kir1:True:Kir2:True:Kir3:True:Kir4:True:Kir7:True:Kir8:True:Kir9:True:Kir10" gainpanel=a | Xtpen pixmaps=y &

# dat/S00243-train5-win1.h5
# dat/S000267-train5-win1.h5
# dat/S000271-train5-win1.h5
# dat/S00275-train5-win1.h5
# dat/S00286-train5-win1.h5
# dat/S00287-train5-win1.h5
# dat/S00295-train5-win1.h5
# dat/S00292-train5-win1.h5


# Histograms
histo:
	Transp plane=12 < dat/S00243-train5-win1.h5_tmax_train.H > t1.H
	Transp plane=12 < dat/S000267-train5-win1.h5_tmax_train.H > t2.H
	Transp plane=12 < dat/S000271-train5-win1.h5_tmax_train.H > t3.H
	Transp plane=12 < dat/S00275-train5-win1.h5_tmax_train.H > t4.H
	Transp plane=12 < dat/S00286-train5-win1.h5_tmax_train.H > t5.H
	Transp plane=12 < dat/S00287-train5-win1.h5_tmax_train.H > t6.H
	Transp plane=12 < dat/S00295-train5-win1.h5_tmax_train.H > t7.H
	Transp plane=12 < dat/S00292-train5-win1.h5_tmax_train.H > t8.H
	# Dev
	Transp plane=12 < dat/S00293-train5-win1.h5_tmax_train.H > d1.H
	Transp plane=12 < dat/S00289-train5-win1.h5_tmax_train.H > d2.H
	Transp plane=12 < dat/S00288-train5-win1.h5_tmax_train.H > d3.H
	Cat axis=1 t1.H t2.H t3.H t4.H t5.H t6.H t7.H t8.H > tTrain.H
	Cat axis=1 d1.H > tDev1.H
	Cat axis=1 d2.H > tDev2.H
	Cat axis=1 d3.H > tDev3.H
	Histogram < tTrain.H | Scale > pTrain.H
	Histogram < tDev1.H | Scale > pDev1.H
	Histogram < tDev2.H | Scale > pDev2.H
	Histogram < tDev3.H | Scale > pDev3.H
	Cat axis=2 pTrain.H pDev1.H pDev2.H pDev3.H | Graph legend=y curvelabel="Train:Dev1:Dev2:Dev3" | Xtpen &
