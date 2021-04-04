################################################################################
################################# Training subset 4 ############################
################################################################################
# Training/dev/test on one full slice
train4_file=par/train_file4.txt
dev4_file=par/dev_file4.txt
n_epoch4=350

################################## Baseline ####################################
train4-win1-base1:
	rm -rf models/train4-win1-base1
	python ./python/CTP_main.py train train4-win1-base1 --model baseline --n_epochs 100 --lr=0.1 --device cuda:0 --train_file_list ${train4_file} --dev_file_list ${dev4_file} --baseline_n_hidden 100

################################## FC Deep #####################################
# MSE + Adam + No lr decay
train4-win1-fcDeep1:
	rm -rf models/train4-win1-fcDeep1
	python ./python/CTP_main.py train train4-win1-fcDeep1 --model fc6 --n_epochs 1000 --lr=0.001 --device cuda:0 --train_file_list ${train4_file} --dev_file_list ${dev4_file}

train4-win1-fcDeep2:
	rm -rf models/train4-win1-fcDeep2
	python ./python/CTP_main.py train train4-win1-fcDeep2 --model fc6 --n_epochs 1000 --lr=0.0005 --device cuda:1 --train_file_list ${train4_file} --dev_file_list ${dev4_file}

train4-win1-fcDeep3:
	rm -rf models/train4-win1-fcDeep3
	python ./python/CTP_main.py train train4-win1-fcDeep3 --model fc6 --n_epochs 1000 --lr=0.003 --device cuda:2 --train_file_list ${train4_file} --dev_file_list ${dev4_file}

train4-win1-fcDeep4:
	rm -rf models/train4-win1-fcDeep4
	python ./python/CTP_main.py train train4-win1-fcDeep4 --model fc6 --n_epochs 2000 --lr=0.0001 --device cuda:3 --train_file_list ${train4_file} --dev_file_list ${dev4_file} --batch_size 1024

train4-win1-fcDeep9:
	rm -rf models/train4-win1-fcDeep9
	python ./python/CTP_main.py train train4-win1-fcDeep9 --model fc6 --n_epochs 50 --lr=0.0005 --device cuda:0 --train_file_list ${train4_file} --dev_file_list ${dev4_file}

train4-win1-fcDeep10:
	rm -rf models/train4-win1-fcDeep10
	python ./python/CTP_main.py train train4-win1-fcDeep10 --model fc6 --n_epochs 100 --lr=0.0001 --device cuda:1 --train_file_list ${train4_file} --dev_file_list ${dev4_file} --batch_size 1024

# MSE + Adam + lr decay
train4-win1-fcDeep5:
	rm -rf models/train4-win1-fcDeep5
	python ./python/CTP_main.py train train4-win1-fcDeep5 --model fc6 --n_epochs 1000 --lr=0.001 --lr_decay step --step_size 100 --decay_gamma 0.99 --device cuda:0 --train_file_list ${train4_file} --dev_file_list ${dev4_file} --batch_size 1024

train4-win1-fcDeep6:
	rm -rf models/train4-win1-fcDeep6
	python ./python/CTP_main.py train train4-win1-fcDeep6 --model fc6 --n_epochs 1000 --lr=0.0005 --lr_decay step --step_size 100 --decay_gamma 0.99 --device cuda:1 --train_file_list ${train4_file} --dev_file_list ${dev4_file} --batch_size 1024

train4-win1-fcDeep7:
	rm -rf models/train4-win1-fcDeep7
	python ./python/CTP_main.py train train4-win1-fcDeep7 --model fc6 --n_epochs 1000 --lr=0.003 --lr_decay step --step_size 100 --decay_gamma 0.99 --device cuda:2 --train_file_list ${train4_file} --dev_file_list ${dev4_file} --batch_size 1024

train4-win1-fcDeep8:
	rm -rf models/train4-win1-fcDeep8
	python ./python/CTP_main.py train train4-win1-fcDeep8 --model fc6 --n_epochs 1000 --lr=0.0001 --lr_decay step --step_size 100 --decay_gamma 0.99 --device cuda:3 --train_file_list ${train4_file} --dev_file_list ${dev4_file} --batch_size 1024

train4-win1-fcDeep11:
	rm -rf models/train4-win1-fcDeep11
	python ./python/CTP_main.py train train4-win1-fcDeep11 --model fc6 --n_epochs 100 --lr=0.003 --lr_decay step --step_size 100 --decay_gamma 0.99 --device cuda:2 --train_file_list ${train4_file} --dev_file_list ${dev4_file} --batch_size 1024

train4-win1-fcDeep12:
	rm -rf models/train4-win1-fcDeep12
	python ./python/CTP_main.py train train4-win1-fcDeep12 --model fc6 --n_epochs 150 --lr=0.0001 --lr_decay step --step_size 100 --decay_gamma 0.99 --device cuda:3 --train_file_list ${train4_file} --dev_file_list ${dev4_file} --batch_size 1024

# Huber + Adam + No lr decay
train4-win1-fcDeep13:
	rm -rf models/train4-win1-fcDeep13
	python ./python/CTP_main.py train train4-win1-fcDeep13 --model fc6 --n_epochs 1000 --lr=0.0001 --loss huber --device cuda:0 --train_file_list ${train4_file} --dev_file_list ${dev4_file}

train4-win1-fcDeep14:
	rm -rf models/train4-win1-fcDeep14
	python ./python/CTP_main.py train train4-win1-fcDeep14 --model fc6 --n_epochs 1000 --lr=0.001 --loss huber --device cuda:1 --train_file_list ${train4_file} --dev_file_list ${dev4_file}

train4-win1-fcDeep15:
	rm -rf models/train4-win1-fcDeep15
	python ./python/CTP_main.py train train4-win1-fcDeep15 --model fc6 --n_epochs 1000 --lr=0.005 --loss huber --device cuda:2 --train_file_list ${train4_file} --dev_file_list ${dev4_file}

train4-win1-fcDeep16:
	rm -rf models/train4-win1-fcDeep16
	python ./python/CTP_main.py train train4-win1-fcDeep16 --model fc6 --n_epochs 1000 --lr=0.005 --loss huber --device cuda:3 --train_file_list ${train4_file} --dev_file_list ${dev4_file}

train4-win1-fcDeep17:
	rm -rf models/train4-win1-fcDeep17
	python ./python/CTP_main.py train train4-win1-fcDeep17 --model fc6 --n_epochs 200 --lr=0.0001 --loss huber --lr_decay step --step_size 50 --decay_gamma 0.99 --device cuda:0 --train_file_list ${train4_file} --dev_file_list ${dev4_file}

train4-win1-fcDeep18:
	rm -rf models/train4-win1-fcDeep18
	python ./python/CTP_main.py train train4-win1-fcDeep18 --model fc6 --n_epochs 200 --lr=0.001 --loss huber --lr_decay step --step_size 50 --decay_gamma 0.99 --device cuda:1 --train_file_list ${train4_file} --dev_file_list ${dev4_file}

train4-win1-fcDeep19:
	rm -rf models/train4-win1-fcDeep19
	python ./python/CTP_main.py train train4-win1-fcDeep19 --model fc6 --n_epochs 200 --lr=0.01 --loss huber --lr_decay step --step_size 50 --decay_gamma 0.99 --device cuda:2 --train_file_list ${train4_file} --dev_file_list ${dev4_file}

train4-win1-fcDeep20:
	rm -rf models/train4-win1-fcDeep20
	python ./python/CTP_main.py train train4-win1-fcDeep20 --model fc6 --n_epochs 200 --lr=0.005 --loss huber --lr_decay step --step_size 50 --decay_gamma 0.99 --device cuda:3 --train_file_list ${train4_file} --dev_file_list ${dev4_file}

# L2 regularization
train4-win1-fcDeep21:
	rm -rf models/train4-win1-fcDeep21
	python ./python/CTP_main.py train train4-win1-fcDeep21 --model fc6 --n_epochs 100 --lr=0.0001 --device cuda:1 --train_file_list ${train4_file} --dev_file_list ${dev4_file} --l2_reg_lambda 0.01 --batch_size 1024

# train4-win1-fcDeep22:
# 	rm -rf models/train4-win1-fcDeep22
# 	python ./python/CTP_main.py train train4-win1-fcDeep22 --model fc6 --n_epochs 3000 --lr=0.001 --loss huber --device cuda:2 --train_file_list ${train4_file} --dev_file_list ${dev4_file}

# L2 regularization
train4-win1-fcDeep23:
	rm -rf models/train4-win1-fcDeep23
	python ./python/CTP_main.py train train4-win1-fcDeep23 --model fc6 --n_epochs 100 --lr=0.0001 --device cuda:0 --train_file_list ${train4_file} --dev_file_list ${dev4_file} --batch_size 1024

train4-win1-fcDeep24:
	rm -rf models/train4-win1-fcDeep24
	python ./python/CTP_main.py train train4-win1-fcDeep24 --model fc6 --n_epochs 100 --lr=0.0001 --device cuda:0 --train_file_list ${train4_file} --dev_file_list ${dev4_file} --l2_reg_lambda 0.001 --batch_size 1024

train4-win1-fcDeep25:
	rm -rf models/train4-win1-fcDeep25
	python ./python/CTP_main.py train train4-win1-fcDeep25 --model fc6 --n_epochs 100 --lr=0.0001 --device cuda:1 --train_file_list ${train4_file} --dev_file_list ${dev4_file} --l2_reg_lambda 0.0001 --batch_size 1024

train4-win1-fcDeep26:
	rm -rf models/train4-win1-fcDeep26
	python ./python/CTP_main.py train train4-win1-fcDeep26 --model fc6 --n_epochs 100 --lr=0.0001 --device cuda:1 --train_file_list ${train4_file} --dev_file_list ${dev4_file} --l2_reg_lambda 0.1 --batch_size 1024

train4-win1-fcDeep27:
	rm -rf models/train4-win1-fcDeep27
	python ./python/CTP_main.py train train4-win1-fcDeep27 --model fc6 --n_epochs 100 --lr=0.0001 --device cuda:3 --train_file_list ${train4_file} --dev_file_list ${dev4_file} --l2_reg_lambda 1.0 --batch_size 1024

train4-win1-fcDeep28:
	rm -rf models/train4-win1-fcDeep28
	python ./python/CTP_main.py train train4-win1-fcDeep28 --model fc6 --n_epochs 100 --lr=0.0001 --device cuda:3 --train_file_list ${train4_file} --dev_file_list ${dev4_file} --l2_reg_lambda 10.0 --batch_size 1024

train4-win1-fcDeep29:
	rm -rf models/train4-win1-fcDeep29
	python ./python/CTP_main.py train train4-win1-fcDeep29 --model fc6 --n_epochs 100 --lr=0.0001 --device cuda:3 --train_file_list ${train4_file} --dev_file_list ${dev4_file} --l2_reg_lambda 100.0 --batch_size 1024

################################### Kir ########################################
# MSE + Adam + No lr decay
train4-win1-kir1:
	rm -rf models/train4-win1-kir1
	python ./python/CTP_main.py train train4-win1-kir1 --model kiranyaz --n_epochs 500 --lr=0.0001 --device cuda:0 --train_file_list ${train4_file} --dev_file_list ${dev4_file}

train4-win1-kir2:
	rm -rf models/train4-win1-kir2
	python ./python/CTP_main.py train train4-win1-kir2 --model kiranyaz --n_epochs 500 --lr=0.001 --device cuda:1 --train_file_list ${train4_file} --dev_file_list ${dev4_file}

train4-win1-kir3:
	rm -rf models/train4-win1-kir3
	python ./python/CTP_main.py train train4-win1-kir3 --model kiranyaz --n_epochs 500 --lr=0.01 --device cuda:2 --train_file_list ${train4_file} --dev_file_list ${dev4_file}

train4-win1-kir4:
	rm -rf models/train4-win1-kir4
	python ./python/CTP_main.py train train4-win1-kir4 --model kiranyaz --n_epochs 500 --lr=0.1 --device cuda:3 --train_file_list ${train4_file} --dev_file_list ${dev4_file}

# MSE + Adam + lr decay
train4-win1-kir5:
	rm -rf models/train4-win1-kir5
	python ./python/CTP_main.py train train4-win1-kir5 --model kiranyaz --n_epochs 500 --lr=0.0001 --lr_decay step --step_size 50 --decay_gamma 0.99 --device cuda:0 --train_file_list ${train4_file} --dev_file_list ${dev4_file}

train4-win1-kir6:
	rm -rf models/train4-win1-kir6
	python ./python/CTP_main.py train train4-win1-kir6 --model kiranyaz --n_epochs 500 --lr=0.001 --lr_decay step --step_size 50 --decay_gamma 0.99 --device cuda:1 --train_file_list ${train4_file} --dev_file_list ${dev4_file}

train4-win1-kir7:
	rm -rf models/train4-win1-kir7
	python ./python/CTP_main.py train train4-win1-kir7 --model kiranyaz --n_epochs 500 --lr=0.01 --lr_decay step --step_size 50 --decay_gamma 0.99 --device cuda:2 --train_file_list ${train4_file} --dev_file_list ${dev4_file}

train4-win1-kir8:
	rm -rf models/train4-win1-kir8
	python ./python/CTP_main.py train train4-win1-kir8 --model kiranyaz --n_epochs 500 --lr=0.05 --lr_decay step --step_size 25 --decay_gamma 0.99 --device cuda:3 --train_file_list ${train4_file} --dev_file_list ${dev4_file}

train4-win1-kir9:
	rm -rf models/train4-win1-kir9
	python ./python/CTP_main.py train train4-win1-kir9 --model kiranyaz --n_epochs 500 --lr=0.00005 --lr_decay step --step_size 50 --decay_gamma 0.99 --device cuda:0 --train_file_list ${train4_file} --dev_file_list ${dev4_file}

train4-win1-kir10:
	rm -rf models/train4-win1-kir10
	python ./python/CTP_main.py train train4-win1-kir10 --model kiranyaz --n_epochs 500 --lr=0.0001 --lr_decay decay --decay_rate 0.001 --device cuda:1 --train_file_list ${train4_file} --dev_file_list ${dev4_file}

train4-win1-kir11:
	rm -rf models/train4-win1-kir11
	python ./python/CTP_main.py train train4-win1-kir11 --optim sgd --model kiranyaz --n_epochs 100 --lr=0.0001 --lr_decay decay --decay_rate 0.001 --device cuda:2 --train_file_list ${train4_file} --dev_file_list ${dev4_file}

train4-win1-kir12:
	rm -rf models/train4-win1-kir12
	python ./python/CTP_main.py train train4-win1-kir12 --optim RMSprop --model kiranyaz --n_epochs 100 --lr=0.0001 --lr_decay decay --decay_rate 0.001 --device cuda:3 --train_file_list ${train4_file} --dev_file_list ${dev4_file}

# Huber + Adam + lr decay
train4-win1-kir13:
	rm -rf models/train4-win1-kir13
	python ./python/CTP_main.py train train4-win1-kir13 --model kiranyaz --n_epochs 100 --lr=0.0001 --loss huber --device cuda:0 --train_file_list ${train4_file} --dev_file_list ${dev4_file}

train4-win1-kir14:
	rm -rf models/train4-win1-kir14
	python ./python/CTP_main.py train train4-win1-kir14 --model kiranyaz --n_epochs 100 --lr=0.0005 --loss huber --device cuda:1 --train_file_list ${train4_file} --dev_file_list ${dev4_file}

train4-win1-kir15:
	rm -rf models/train4-win1-kir15
	python ./python/CTP_main.py train train4-win1-kir15 --model kiranyaz --n_epochs 100 --lr=0.001 --loss huber --device cuda:2 --train_file_list ${train4_file} --dev_file_list ${dev4_file}

train4-win1-kir16:
	rm -rf models/train4-win1-kir16
	python ./python/CTP_main.py train train4-win1-kir16 --model kiranyaz --n_epochs 100 --lr=0.005 --loss huber --device cuda:3 --train_file_list ${train4_file} --dev_file_list ${dev4_file}

train4-win1-kir17:
	rm -rf models/train4-win1-kir17
	python ./python/CTP_main.py train train4-win1-kir17 --model kiranyaz --n_epochs 100 --lr=0.005 --loss huber --lr_decay decay --decay_rate 0.001 --device cuda:3 --train_file_list ${train4_file} --dev_file_list ${dev4_file}

train4-win1-kir18:
	rm -rf models/train4-win1-kir18
	python ./python/CTP_main.py train train4-win1-kir18 --model kiranyaz --n_epochs 100 --lr=0.008 --loss huber --lr_decay decay --decay_rate 0.001 --device cuda:2 --train_file_list ${train4_file} --dev_file_list ${dev4_file}

train4-win1-kir19:
	rm -rf models/train4-win1-kir19
	python ./python/CTP_main.py train train4-win1-kir19 --model kiranyaz --n_epochs 100 --lr=0.005 --loss huber --lr_decay step --step_size 50 --decay_gamma 0.99 --device cuda:0 --train_file_list ${train4_file} --dev_file_list ${dev4_file}

train4-win1-kir20:
	rm -rf models/train4-win1-kir20
	python ./python/CTP_main.py train train4-win1-kir20 --model kiranyaz --n_epochs 100 --lr=0.001 --loss huber --lr_decay step --step_size 50 --decay_gamma 0.99 --device cuda:1 --train_file_list ${train4_file} --dev_file_list ${dev4_file}

train4-win1-kir21:
	rm -rf models/train4-win1-kir21
	python ./python/CTP_main.py train train4-win1-kir21 --model kiranyaz --n_epochs 200 --lr=0.001 --loss huber --lr_decay step --step_size 50 --decay_gamma 0.99 --device cuda:0 --train_file_list ${train4_file} --dev_file_list ${dev4_file}

train4-win1-kir22:
	rm -rf models/train4-win1-kir22
	python ./python/CTP_main.py train train4-win1-kir22 --model kiranyaz --n_epochs 200 --lr=0.001 --loss huber --lr_decay step --step_size 50 --decay_gamma 0.99 --device cpu --train_file_list ${train4_file} --dev_file_list ${dev4_file}

################################### Blog med ###################################
# MSE + Adam + No lr decay
train4-win1-blog1:
	rm -rf models/train4-win1-blog1
	CUDA_LAUNCH_BLOCKING=1 python ./python/CTP_main.py train train4-win1-blog1 --model blog --n_epochs 100 --lr=0.00001 --device cuda:0 --train_file_list ${train4_file} --dev_file_list ${dev4_file}

train4-win1-blog-test1:
	rm -rf models/train4-win1-blog1-test1
	python ./python/CTP_main.py train train4-win1-blog1-test1 --model blog --n_epochs 100 --lr=0.001 --device cuda:1 --train_file_list ${train4_file} --dev_file_list ${dev4_file}

train4-win1-blog-test2:
	rm -rf models/train4-win1-blog1-test2
	CUDA_LAUNCH_BLOCKING=1 python ./python/CTP_main.py train train4-win1-blog1-test2 --model blog_test --n_epochs 100 --lr=0.00001 --device cuda:1 --train_file_list ${train4_file} --dev_file_list ${dev4_file}

train4-win1-blog-test3:
	rm -rf models/train4-win1-blog1-test3
	CUDA_LAUNCH_BLOCKING=1 python ./python/CTP_main.py train train4-win1-blog1-test3 --model kiranyaz --n_epochs 100 --lr=0.00001 --device cuda:2 --train_file_list ${train4_file} --dev_file_list ${dev4_file}

train4-win1-blog2:
	rm -rf models/train4-win1-blog2
	python ./python/CTP_main.py train train4-win1-blog2 --model blog --n_epochs 100 --lr=0.0001 --device cuda:1 --train_file_list ${train4_file} --dev_file_list ${dev4_file}

train4-win1-blog3:
	rm -rf models/train4-win1-blog3
	python ./python/CTP_main.py train train4-win1-blog3 --model blog --n_epochs 100 --lr=0.001 --device cuda:2 --train_file_list ${train4_file} --dev_file_list ${dev4_file}

train4-win1-blog4:
	rm -rf models/train4-win1-blog4
	python ./python/CTP_main.py train train4-win1-blog4 --model blog --n_epochs 100 --lr=0.01 --device cuda:3 --train_file_list ${train4_file} --dev_file_list ${dev4_file}

train4-win1-blog5:
	rm -rf models/train4-win1-blog5
	python ./python/CTP_main.py train train4-win1-blog5 --model blog --n_epochs 1000 --lr=0.0005 --device cuda:0 --train_file_list ${train4_file} --dev_file_list ${dev4_file}

train4-win1-blog6:
	rm -rf models/train4-win1-blog6
	python ./python/CTP_main.py train train4-win1-blog6 --model blog --n_epochs 1000 --lr=0.0005 --lr_decay step --step_size 50 --decay_gamma 0.99 --device cuda:1 --train_file_list ${train4_file} --dev_file_list ${dev4_file}

train4-win1-blog7:
	rm -rf models/train4-win1-blog7
	python ./python/CTP_main.py train train4-win1-blog7 --model blog --n_epochs 125 --lr=0.0001 --lr_decay step --step_size 50 --decay_gamma 0.99 --device cuda:2 --train_file_list ${train4_file} --dev_file_list ${dev4_file}

train4-win1-blog7-test:
	rm -rf models/train4-win1-blog7-test
	python ./python/CTP_main.py train train4-win1-blog7-test --model blog --n_epochs 125 --lr=0.0001 --lr_decay step --step_size 50 --decay_gamma 0.99 --device cuda:0 --train_file_list ${train4_file} --dev_file_list ${dev4_file}

train4-win1-blog8:
	rm -rf models/train4-win1-blog8
	python ./python/CTP_main.py train train4-win1-blog8 --model blog --n_epochs 1000 --lr=0.0001 --lr_decay decay --decay_rate 0.001 --device cuda:3 --train_file_list ${train4_file} --dev_file_list ${dev4_file}

# Huber + Adam + lr decay
train4-win1-blog9:
	rm -rf models/train4-win1-blog9
	python ./python/CTP_main.py train train4-win1-blog9 --model blog --n_epochs 1000 --loss huber --lr=0.0001 --device cuda:0 --train_file_list ${train4_file} --dev_file_list ${dev4_file}

train4-win1-blog10:
	rm -rf models/train4-win1-blog10
	python ./python/CTP_main.py train train4-win1-blog10 --model blog --n_epochs 1000 --loss huber --lr=0.001 --device cuda:1 --train_file_list ${train4_file} --dev_file_list ${dev4_file}

train4-win1-blog11:
	rm -rf models/train4-win1-blog11
	python ./python/CTP_main.py train train4-win1-blog11 --model blog --n_epochs 1000 --loss huber --lr=0.01 --device cuda:2 --train_file_list ${train4_file} --dev_file_list ${dev4_file}

train4-win1-blog12:
	rm -rf models/train4-win1-blog12
	python ./python/CTP_main.py train train4-win1-blog12 --model blog --n_epochs 1000 --loss huber --lr=0.005 --device cuda:0 --train_file_list ${train4_file} --dev_file_list ${dev4_file}

################################################################################
############################ Prediction on train data ##########################
################################################################################
pred4_file_train=dat/S00243-train4-win1.h5
# pred4_file_train=dat/S000267-train4-win1.h5
# pred4_file_train=dat/S00295-train4-win1.h5
# pred4_file_train=dat/S00275-train4-win1.h5

predictTrain4-win1-base%:
	python ./python/CTP_main.py predict train4-win1-base$* --model baseline --patient_file ${pred4_file_train} --patient_id S00243 --baseline_n_hidden 100

predictTrain4-win1-fcDeep%:
	python ./python/CTP_main.py predict train4-win1-fcDeep$* --model fc6 --device cuda:0 --patient_file ${pred4_file_train} --patient_id S00243

predictTrain4-win1-kir%:
	python ./python/CTP_main.py predict train4-win1-kir$* --model kiranyaz --device cuda:0 --patient_file ${pred4_file_train} --patient_id S00243

predictTrain4-win1-blog%:
	python ./python/CTP_main.py predict train4-win1-blog$* --model blog --device cuda:0 --patient_file ${pred4_file_train} --patient_id S00243

# predictTrain4-win1-blogDeep%:
# 	python ./python/CTP_main.py predict train4-win1-blogDeep$* --model blogDeep --device cuda:0 --patient_file ${pred4_file_train} --patient_id S00243

makePredTrain4:
	# make predictTrain4-win1-base1 -B
	# make predictTrain4-win1-fcDeep1 -B
	# make predictTrain4-win1-fcDeep2 -B
	# make predictTrain4-win1-fcDeep3 -B
	# make predictTrain4-win1-fcDeep4 -B
	# make predictTrain4-win1-fcDeep5 -B
	# make predictTrain4-win1-fcDeep6 -B
	# make predictTrain4-win1-fcDeep7 -B
	# make predictTrain4-win1-fcDeep8 -B
	# make predictTrain4-win1-fcDeep9 -B
	# make predictTrain4-win1-fcDeep10 -B
	# make predictTrain4-win1-fcDeep11 -B
	# make predictTrain4-win1-fcDeep12 -B
	# make predictTrain4-win1-fcDeep13 -B
	# make predictTrain4-win1-fcDeep14 -B
	# make predictTrain4-win1-fcDeep15 -B
	# make predictTrain4-win1-fcDeep16 -B
	# make predictTrain4-win1-fcDeep17 -B
	# make predictTrain4-win1-fcDeep18 -B
	# make predictTrain4-win1-fcDeep19 -B
	# make predictTrain4-win1-fcDeep20 -B
	# make predictTrain4-win1-fcDeep21 -B
	# make predictTrain4-win1-fcDeep23 -B
	# make predictTrain4-win1-fcDeep24 -B
	# make predictTrain4-win1-fcDeep25 -B
	# make predictTrain4-win1-fcDeep26 -B
	# make predictTrain4-win1-fcDeep27 -B
	make predictTrain4-win1-kir1 -B
	make predictTrain4-win1-kir2 -B
	make predictTrain4-win1-kir3 -B
	make predictTrain4-win1-kir4 -B
	make predictTrain4-win1-kir5 -B
	make predictTrain4-win1-kir6 -B
	make predictTrain4-win1-kir7 -B
	make predictTrain4-win1-kir8 -B
	make predictTrain4-win1-kir9 -B
	make predictTrain4-win1-kir10 -B
	make predictTrain4-win1-kir11 -B
	make predictTrain4-win1-kir12 -B
	make predictTrain4-win1-kir13 -B
	make predictTrain4-win1-kir14 -B
	make predictTrain4-win1-kir15 -B
	make predictTrain4-win1-kir16 -B
	make predictTrain4-win1-kir17 -B
	make predictTrain4-win1-kir18 -B
	make predictTrain4-win1-blog1 -B
	make predictTrain4-win1-blog2 -B
	make predictTrain4-win1-blog3 -B
	make predictTrain4-win1-blog4 -B
	make predictTrain4-win1-blog5 -B
	make predictTrain4-win1-blog6 -B
	make predictTrain4-win1-blog7 -B
	make predictTrain4-win1-blog8 -B
	make predictTrain4-win1-blog9 -B
	make predictTrain4-win1-blog10 -B
	make predictTrain4-win1-blog11 -B
	make predictTrain4-win1-blog12 -B

################################################################################
################################# Display ######################################
################################################################################
# type4=_avg
type4=
eclip4=400
bclip4=0
dispResult4-s%:
	# Training data
	Window3d n3=1 f3=$* < ${pred4_file_train}_tmax_m.H > t1.H
	# Base
	Window3d n3=1 f3=$* < models/train4-win1-base1/train4-win1-base1_S00243_pred_tmax${type4}_sep.H > t2.H
	# FC
	Window3d n3=1 f3=$* < models/train4-win1-fcDeep1/train4-win1-fcDeep1_S00243_pred_tmax${type4}_sep.H > t3a.H
	Window3d n3=1 f3=$* < models/train4-win1-fcDeep2/train4-win1-fcDeep2_S00243_pred_tmax${type4}_sep.H > t3b.H
	Window3d n3=1 f3=$* < models/train4-win1-fcDeep3/train4-win1-fcDeep3_S00243_pred_tmax${type4}_sep.H > t3c.H
	Window3d n3=1 f3=$* < models/train4-win1-fcDeep4/train4-win1-fcDeep4_S00243_pred_tmax${type4}_sep.H > t3d.H
	Window3d n3=1 f3=$* < models/train4-win1-fcDeep5/train4-win1-fcDeep5_S00243_pred_tmax${type4}_sep.H > t3e.H
	Window3d n3=1 f3=$* < models/train4-win1-fcDeep6/train4-win1-fcDeep6_S00243_pred_tmax${type4}_sep.H > t3f.H
	Window3d n3=1 f3=$* < models/train4-win1-fcDeep7/train4-win1-fcDeep7_S00243_pred_tmax${type4}_sep.H > t3g.H
	Window3d n3=1 f3=$* < models/train4-win1-fcDeep8/train4-win1-fcDeep8_S00243_pred_tmax${type4}_sep.H > t3h.H
	Window3d n3=1 f3=$* < models/train4-win1-fcDeep9/train4-win1-fcDeep9_S00243_pred_tmax${type4}_sep.H > t3i.H
	Window3d n3=1 f3=$* < models/train4-win1-fcDeep10/train4-win1-fcDeep10_S00243_pred_tmax${type4}_sep.H > t3j.H
	Window3d n3=1 f3=$* < models/train4-win1-fcDeep11/train4-win1-fcDeep11_S00243_pred_tmax${type4}_sep.H > t3k.H
	Window3d n3=1 f3=$* < models/train4-win1-fcDeep12/train4-win1-fcDeep12_S00243_pred_tmax${type4}_sep.H > t3l.H
	Window3d n3=1 f3=$* < models/train4-win1-fcDeep13/train4-win1-fcDeep13_S00243_pred_tmax${type4}_sep.H > t3m.H
	Window3d n3=1 f3=$* < models/train4-win1-fcDeep14/train4-win1-fcDeep14_S00243_pred_tmax${type4}_sep.H > t3n.H
	Window3d n3=1 f3=$* < models/train4-win1-fcDeep15/train4-win1-fcDeep15_S00243_pred_tmax${type4}_sep.H > t3o.H
	Window3d n3=1 f3=$* < models/train4-win1-fcDeep16/train4-win1-fcDeep16_S00243_pred_tmax${type4}_sep.H > t3p.H
	Window3d n3=1 f3=$* < models/train4-win1-fcDeep17/train4-win1-fcDeep17_S00243_pred_tmax${type4}_sep.H > t3q.H
	Window3d n3=1 f3=$* < models/train4-win1-fcDeep18/train4-win1-fcDeep18_S00243_pred_tmax${type4}_sep.H > t3r.H
	Window3d n3=1 f3=$* < models/train4-win1-fcDeep19/train4-win1-fcDeep19_S00243_pred_tmax${type4}_sep.H > t3s.H
	# Kir
	Window3d n3=1 f3=$* < models/train4-win1-kir1/train4-win1-kir1_S00243_pred_tmax${type4}_sep.H > t4a.H
	Window3d n3=1 f3=$* < models/train4-win1-kir2/train4-win1-kir2_S00243_pred_tmax${type4}_sep.H > t4b.H
	Window3d n3=1 f3=$* < models/train4-win1-kir3/train4-win1-kir3_S00243_pred_tmax${type4}_sep.H > t4c.H
	Window3d n3=1 f3=$* < models/train4-win1-kir4/train4-win1-kir4_S00243_pred_tmax${type4}_sep.H > t4d.H
	Window3d n3=1 f3=$* < models/train4-win1-kir5/train4-win1-kir5_S00243_pred_tmax${type4}_sep.H > t4d.H
	Window3d n3=1 f3=$* < models/train4-win1-kir6/train4-win1-kir6_S00243_pred_tmax${type4}_sep.H > t4e.H
	Window3d n3=1 f3=$* < models/train4-win1-kir7/train4-win1-kir7_S00243_pred_tmax${type4}_sep.H > t4f.H
	Window3d n3=1 f3=$* < models/train4-win1-kir8/train4-win1-kir8_S00243_pred_tmax${type4}_sep.H > t4g.H
	Window3d n3=1 f3=$* < models/train4-win1-kir9/train4-win1-kir9_S00243_pred_tmax${type4}_sep.H > t4h.H
	Window3d n3=1 f3=$* < models/train4-win1-kir10/train4-win1-kir10_S00243_pred_tmax${type4}_sep.H > t4i.H
	Window3d n3=1 f3=$* < models/train4-win1-kir11/train4-win1-kir11_S00243_pred_tmax${type4}_sep.H > t4j.H
	Window3d n3=1 f3=$* < models/train4-win1-kir12/train4-win1-kir12_S00243_pred_tmax${type4}_sep.H > t4k.H
	Window3d n3=1 f3=$* < models/train4-win1-kir13/train4-win1-kir13_S00243_pred_tmax${type4}_sep.H > t4l.H
	Window3d n3=1 f3=$* < models/train4-win1-kir14/train4-win1-kir14_S00243_pred_tmax${type4}_sep.H > t4m.H
	Window3d n3=1 f3=$* < models/train4-win1-kir15/train4-win1-kir15_S00243_pred_tmax${type4}_sep.H > t4n.H
	Window3d n3=1 f3=$* < models/train4-win1-kir16/train4-win1-kir16_S00243_pred_tmax${type4}_sep.H > t4o.H
	Window3d n3=1 f3=$* < models/train4-win1-kir17/train4-win1-kir17_S00243_pred_tmax${type4}_sep.H > t4p.H
	Window3d n3=1 f3=$* < models/train4-win1-kir18/train4-win1-kir18_S00243_pred_tmax${type4}_sep.H > t4q.H
	Window3d n3=1 f3=$* < models/train4-win1-blog1/train4-win1-blog1_S00243_pred_tmax${type4}_sep.H > t5a.H
	Window3d n3=1 f3=$* < models/train4-win1-blog2/train4-win1-blog2_S00243_pred_tmax${type4}_sep.H > t5b.H
	Window3d n3=1 f3=$* < models/train4-win1-blog3/train4-win1-blog3_S00243_pred_tmax${type4}_sep.H > t5c.H
	Window3d n3=1 f3=$* < models/train4-win1-blog4/train4-win1-blog4_S00243_pred_tmax${type4}_sep.H > t5d.H
	Window3d n3=1 f3=$* < models/train4-win1-blog5/train4-win1-blog5_S00243_pred_tmax${type4}_sep.H > t5e.H
	Window3d n3=1 f3=$* < models/train4-win1-blog6/train4-win1-blog6_S00243_pred_tmax${type4}_sep.H > t5f.H
	Window3d n3=1 f3=$* < models/train4-win1-blog7/train4-win1-blog7_S00243_pred_tmax${type4}_sep.H > t5g.H
	Window3d n3=1 f3=$* < models/train4-win1-blog8/train4-win1-blog8_S00243_pred_tmax${type4}_sep.H > t5h.H
	Window3d n3=1 f3=$* < models/train4-win1-blog9/train4-win1-blog9_S00243_pred_tmax${type4}_sep.H > t5i.H
	Window3d n3=1 f3=$* < models/train4-win1-blog10/train4-win1-blog10_S00243_pred_tmax${type4}_sep.H > t5j.H
	Window3d n3=1 f3=$* < models/train4-win1-blog11/train4-win1-blog11_S00243_pred_tmax${type4}_sep.H > t5k.H
	Window3d n3=1 f3=$* < models/train4-win1-blog12/train4-win1-blog12_S00243_pred_tmax${type4}_sep.H > t5l.H
	# Display FC
	Cat axis=3 t1.H t2.H t1.H t3a.H t1.H t3b.H t1.H t3c.H t1.H t3d.H t1.H t3e.H t1.H t3f.H t1.H t3g.H t1.H t3h.H t1.H t3i.H t1.H t3j.H t1.H t3k.H t1.H t3l.H | Grey color=j newclip=1 bclip=${bclip4} eclip=${eclip4} grid=y titles="True:Base:True:Fc1:True:Fc2:True:Fc3:True:Fc4:True:Fc5:True:Fc6:True:Fc7:True:Fc8:True:Fc9:True:Fc10:True:Fc11:True:Fc12" gainpanel=a | Xtpen pixmaps=y &
	# Display FC + Huber
	Cat axis=3 t1.H t3m.H t1.H t3n.H t1.H t3o.H t1.H t3p.H t1.H t3q.H t1.H t3r.H t1.H t3s.H | Grey color=j newclip=1 bclip=${bclip4} eclip=${eclip4} grid=y titles="True:Fc13:True:Fc14:True:Fc15:True:Fc16:True:Fc17:True:Fc17:True:Fc18:True:Fc19:True:Fc20" gainpanel=a | Xtpen pixmaps=y &
	# Display Kir
	# Cat axis=3 t1.H t4a.H t1.H t4b.H t1.H t4c.H t1.H t4d.H t1.H t4e.H t1.H t4f.H t1.H t4g.H t1.H t4h.H t1.H t4i.H t1.H t4j.H t1.H t4k.H | Grey color=j newclip=1 bclip=${bclip4} eclip=${eclip4} grid=y titles="True:Kir1:True:Kir2:True:Kir3:True:Kir4:True:Kir5:True:Kir6:True:Kir7:True:Kir8:True:Kir9:True:Kir10:True:Kir11:True:Kir12" gainpanel=a | Xtpen pixmaps=y &
	# # Display Kir + Huber
	# Cat axis=3 t1.H t4l.H t1.H t4m.H t1.H t4m.H t1.H t4n.H t1.H t4o.H t1.H t4p.H t1.H t4q.H | Grey color=j newclip=1 bclip=${bclip4} eclip=${eclip4} grid=y titles="True:Kir13:True:Kir14:True:Kir15:True:Kir16:Tru:Kir17:True:Kir18" gainpanel=a | Xtpen pixmaps=y &
	# # Display Blog
	# Cat axis=3 t1.H t5a.H t1.H t5b.H t1.H t5c.H t1.H t5d.H t1.H t5e.H t1.H t5f.H t1.H t5g.H t1.H t5h.H t1.H t5i.H t1.H t5j.H t1.H t5k.H t1.H t5l.H | Grey color=j newclip=1 bclip=${bclip4} eclip=${eclip4} grid=y titles="True:Blog1:True:Blog2:True:Blog3:True:Blog4:True:Blog5:True:Blog6:True:Blog7:True:Blog8:True:Blog9:True:Blog10:True:Blog11:True:Blog12" gainpanel=a | Xtpen pixmaps=y &

################################################################################
############################ Prediction on dev data ############################
################################################################################
pred4_file_dev=dat/S00287-train4-win1.h5

predictDev4-win1-base%:
	python ./python/CTP_main.py predict train4-win1-base$* --model baseline --patient_file ${pred4_file_dev} --patient_id S00243d --baseline_n_hidden 100

predictDev4-win1-fcDeep%:
	python ./python/CTP_main.py predict train4-win1-fcDeep$* --model fc6 --device cuda:0 --patient_file ${pred4_file_dev} --patient_id S00243d

predictDev4-win1-kir%:
	python ./python/CTP_main.py predict train4-win1-kir$* --model kiranyaz --device cuda:0 --patient_file ${pred4_file_dev} --patient_id S00243d

predictDev4-win1-blog%:
	python ./python/CTP_main.py predict train4-win1-blog$* --model blog --device cuda:0 --patient_file ${pred4_file_dev} --patient_id S00243d

predictDev4-win1-blogDeep%:
	python ./python/CTP_main.py predict train4-win1-blogDeep$* --model blogDeep --device cuda:0 --patient_file ${pred4_file_dev} --patient_id S00243d

makePredDev4:
	# make predictDev4-win1-base1 -B
	# make predictDev4-win1-fcDeep1 -B
	# make predictDev4-win1-fcDeep2 -B
	# make predictDev4-win1-fcDeep3 -B
	# make predictDev4-win1-fcDeep4 -B
	# make predictDev4-win1-fcDeep5 -B
	# make predictDev4-win1-fcDeep6 -B
	# make predictDev4-win1-fcDeep7 -B
	# make predictDev4-win1-fcDeep8 -B
	# make predictDev4-win1-fcDeep9 -B
	# make predictDev4-win1-fcDeep10 -B
	# make predictDev4-win1-fcDeep11 -B
	# make predictDev4-win1-fcDeep12 -B
	# make predictDev4-win1-fcDeep13 -B
	# make predictDev4-win1-fcDeep14 -B
	# make predictDev4-win1-fcDeep15 -B
	# make predictDev4-win1-fcDeep16 -B
	# make predictDev4-win1-fcDeep17 -B
	# make predictDev4-win1-fcDeep18 -B
	# make predictDev4-win1-fcDeep19 -B
	# make predictDev4-win1-fcDeep20 -B
	# make predictDev4-win1-fcDeep21 -B
	# make predictDev4-win1-fcDeep23 -B
	# make predictDev4-win1-fcDeep24 -B
	# make predictDev4-win1-fcDeep25 -B
	# make predictDev4-win1-fcDeep26 -B
	# make predictDev4-win1-fcDeep27 -B
	make predictDev4-win1-kir1 -B
	make predictDev4-win1-kir2 -B
	make predictDev4-win1-kir3 -B
	make predictDev4-win1-kir4 -B
	make predictDev4-win1-kir5 -B
	make predictDev4-win1-kir6 -B
	make predictDev4-win1-kir7 -B
	make predictDev4-win1-kir8 -B
	make predictDev4-win1-kir9 -B
	make predictDev4-win1-kir10 -B
	make predictDev4-win1-kir11 -B
	make predictDev4-win1-kir12 -B
	make predictDev4-win1-kir13 -B
	make predictDev4-win1-kir14 -B
	make predictDev4-win1-kir15 -B
	make predictDev4-win1-kir16 -B
	make predictDev4-win1-kir17 -B
	make predictDev4-win1-kir18 -B
	make predictDev4-win1-blog1 -B
	make predictDev4-win1-blog2 -B
	make predictDev4-win1-blog3 -B
	make predictDev4-win1-blog4 -B
	make predictDev4-win1-blog5 -B
	make predictDev4-win1-blog6 -B
	make predictDev4-win1-blog7 -B
	make predictDev4-win1-blog8 -B
	make predictDev4-win1-blog9 -B
	make predictDev4-win1-blog10 -B
	make predictDev4-win1-blog11 -B
	make predictDev4-win1-blog12 -B

################################################################################
################################# Display ######################################
################################################################################
dispResultDev4-s%:
	# Training data
	Window3d n3=1 f3=$* < ${pred4_file_dev}_tmax_m.H > t1.H
	# Base
	Window3d n3=1 f3=$* < models/train4-win1-base1/train4-win1-base1_S00243d_pred_tmax${type4}_sep.H > t2.H
	# FC
	Window3d n3=1 f3=$* < models/train4-win1-fcDeep1/train4-win1-fcDeep1_S00243d_pred_tmax${type4}_sep.H > t3a.H
	Window3d n3=1 f3=$* < models/train4-win1-fcDeep2/train4-win1-fcDeep2_S00243d_pred_tmax${type4}_sep.H > t3b.H
	Window3d n3=1 f3=$* < models/train4-win1-fcDeep3/train4-win1-fcDeep3_S00243d_pred_tmax${type4}_sep.H > t3c.H
	Window3d n3=1 f3=$* < models/train4-win1-fcDeep4/train4-win1-fcDeep4_S00243d_pred_tmax${type4}_sep.H > t3d.H
	Window3d n3=1 f3=$* < models/train4-win1-fcDeep5/train4-win1-fcDeep5_S00243d_pred_tmax${type4}_sep.H > t3e.H
	Window3d n3=1 f3=$* < models/train4-win1-fcDeep6/train4-win1-fcDeep6_S00243d_pred_tmax${type4}_sep.H > t3f.H
	Window3d n3=1 f3=$* < models/train4-win1-fcDeep7/train4-win1-fcDeep7_S00243d_pred_tmax${type4}_sep.H > t3g.H
	Window3d n3=1 f3=$* < models/train4-win1-fcDeep8/train4-win1-fcDeep8_S00243d_pred_tmax${type4}_sep.H > t3h.H
	Window3d n3=1 f3=$* < models/train4-win1-fcDeep9/train4-win1-fcDeep9_S00243d_pred_tmax${type4}_sep.H > t3i.H
	Window3d n3=1 f3=$* < models/train4-win1-fcDeep10/train4-win1-fcDeep10_S00243d_pred_tmax${type4}_sep.H > t3j.H
	Window3d n3=1 f3=$* < models/train4-win1-fcDeep11/train4-win1-fcDeep11_S00243d_pred_tmax${type4}_sep.H > t3k.H
	Window3d n3=1 f3=$* < models/train4-win1-fcDeep12/train4-win1-fcDeep12_S00243d_pred_tmax${type4}_sep.H > t3l.H
	Window3d n3=1 f3=$* < models/train4-win1-fcDeep13/train4-win1-fcDeep13_S00243d_pred_tmax${type4}_sep.H > t3m.H
	Window3d n3=1 f3=$* < models/train4-win1-fcDeep14/train4-win1-fcDeep14_S00243d_pred_tmax${type4}_sep.H > t3n.H
	Window3d n3=1 f3=$* < models/train4-win1-fcDeep15/train4-win1-fcDeep15_S00243d_pred_tmax${type4}_sep.H > t3o.H
	Window3d n3=1 f3=$* < models/train4-win1-fcDeep16/train4-win1-fcDeep16_S00243d_pred_tmax${type4}_sep.H > t3p.H
	Window3d n3=1 f3=$* < models/train4-win1-fcDeep17/train4-win1-fcDeep17_S00243d_pred_tmax${type4}_sep.H > t3q.H
	Window3d n3=1 f3=$* < models/train4-win1-fcDeep18/train4-win1-fcDeep18_S00243d_pred_tmax${type4}_sep.H > t3r.H
	Window3d n3=1 f3=$* < models/train4-win1-fcDeep19/train4-win1-fcDeep19_S00243d_pred_tmax${type4}_sep.H > t3s.H
	# Kir
	Window3d n3=1 f3=$* < models/train4-win1-kir1/train4-win1-kir1_S00243d_pred_tmax${type4}_sep.H > t4a.H
	Window3d n3=1 f3=$* < models/train4-win1-kir2/train4-win1-kir2_S00243d_pred_tmax${type4}_sep.H > t4b.H
	Window3d n3=1 f3=$* < models/train4-win1-kir3/train4-win1-kir3_S00243d_pred_tmax${type4}_sep.H > t4c.H
	Window3d n3=1 f3=$* < models/train4-win1-kir4/train4-win1-kir4_S00243d_pred_tmax${type4}_sep.H > t4d.H
	Window3d n3=1 f3=$* < models/train4-win1-kir5/train4-win1-kir5_S00243d_pred_tmax${type4}_sep.H > t4d.H
	Window3d n3=1 f3=$* < models/train4-win1-kir6/train4-win1-kir6_S00243d_pred_tmax${type4}_sep.H > t4e.H
	Window3d n3=1 f3=$* < models/train4-win1-kir7/train4-win1-kir7_S00243d_pred_tmax${type4}_sep.H > t4f.H
	Window3d n3=1 f3=$* < models/train4-win1-kir8/train4-win1-kir8_S00243d_pred_tmax${type4}_sep.H > t4g.H
	Window3d n3=1 f3=$* < models/train4-win1-kir9/train4-win1-kir9_S00243d_pred_tmax${type4}_sep.H > t4h.H
	Window3d n3=1 f3=$* < models/train4-win1-kir10/train4-win1-kir10_S00243d_pred_tmax${type4}_sep.H > t4i.H
	Window3d n3=1 f3=$* < models/train4-win1-kir11/train4-win1-kir11_S00243d_pred_tmax${type4}_sep.H > t4j.H
	Window3d n3=1 f3=$* < models/train4-win1-kir12/train4-win1-kir12_S00243d_pred_tmax${type4}_sep.H > t4k.H
	Window3d n3=1 f3=$* < models/train4-win1-kir13/train4-win1-kir13_S00243d_pred_tmax${type4}_sep.H > t4l.H
	Window3d n3=1 f3=$* < models/train4-win1-kir14/train4-win1-kir14_S00243d_pred_tmax${type4}_sep.H > t4m.H
	Window3d n3=1 f3=$* < models/train4-win1-kir15/train4-win1-kir15_S00243d_pred_tmax${type4}_sep.H > t4n.H
	Window3d n3=1 f3=$* < models/train4-win1-kir16/train4-win1-kir16_S00243d_pred_tmax${type4}_sep.H > t4o.H
	Window3d n3=1 f3=$* < models/train4-win1-kir17/train4-win1-kir17_S00243d_pred_tmax${type4}_sep.H > t4p.H
	Window3d n3=1 f3=$* < models/train4-win1-kir18/train4-win1-kir18_S00243d_pred_tmax${type4}_sep.H > t4q.H
	Window3d n3=1 f3=$* < models/train4-win1-blog1/train4-win1-blog1_S00243d_pred_tmax${type4}_sep.H > t5a.H
	Window3d n3=1 f3=$* < models/train4-win1-blog2/train4-win1-blog2_S00243d_pred_tmax${type4}_sep.H > t5b.H
	Window3d n3=1 f3=$* < models/train4-win1-blog3/train4-win1-blog3_S00243d_pred_tmax${type4}_sep.H > t5c.H
	Window3d n3=1 f3=$* < models/train4-win1-blog4/train4-win1-blog4_S00243d_pred_tmax${type4}_sep.H > t5d.H
	Window3d n3=1 f3=$* < models/train4-win1-blog5/train4-win1-blog5_S00243d_pred_tmax${type4}_sep.H > t5e.H
	Window3d n3=1 f3=$* < models/train4-win1-blog6/train4-win1-blog6_S00243d_pred_tmax${type4}_sep.H > t5f.H
	Window3d n3=1 f3=$* < models/train4-win1-blog7/train4-win1-blog7_S00243d_pred_tmax${type4}_sep.H > t5g.H
	Window3d n3=1 f3=$* < models/train4-win1-blog8/train4-win1-blog8_S00243d_pred_tmax${type4}_sep.H > t5h.H
	Window3d n3=1 f3=$* < models/train4-win1-blog9/train4-win1-blog9_S00243d_pred_tmax${type4}_sep.H > t5i.H
	Window3d n3=1 f3=$* < models/train4-win1-blog10/train4-win1-blog10_S00243d_pred_tmax${type4}_sep.H > t5j.H
	Window3d n3=1 f3=$* < models/train4-win1-blog11/train4-win1-blog11_S00243d_pred_tmax${type4}_sep.H > t5k.H
	Window3d n3=1 f3=$* < models/train4-win1-blog12/train4-win1-blog12_S00243d_pred_tmax${type4}_sep.H > t5l.H
	# Display
	Cat axis=3 t1.H t2.H t1.H t3a.H t1.H t3b.H t1.H t3c.H t1.H t3d.H t1.H t3e.H t1.H t3f.H t1.H t3g.H t1.H t3h.H t1.H t3i.H t1.H t3j.H t1.H t3k.H t1.H t3l.H | Grey color=j newclip=1 bclip=${bclip4} eclip=${eclip4} grid=y titles="True:Base:True:Fc1:True:Fc2:True:Fc3:True:Fc4:True:Fc5:True:Fc6:True:Fc7:True:Fc8:True:Fc9:True:Fc10:True:Fc11:True:Fc12" gainpanel=a | Xtpen pixmaps=y &
	Cat axis=3 t1.H t3m.H t1.H t3n.H t1.H t3o.H t1.H t3p.H t1.H t3q.H t1.H t3r.H t1.H t3s.H | Grey color=j newclip=1 bclip=${bclip4} eclip=${eclip4} grid=y titles="True:Fc13:True:Fc14:True:Fc15:True:Fc16:True:Fc17:True:Fc17:True:Fc18:True:Fc19:True:Fc20" gainpanel=a | Xtpen pixmaps=y &
	Cat axis=3 t1.H t4a.H t1.H t4b.H t1.H t4c.H t1.H t4d.H t1.H t4e.H t1.H t4f.H t1.H t4g.H t1.H t4h.H t1.H t4i.H t1.H t4j.H t1.H t4k.H | Grey color=j newclip=1 bclip=${bclip4} eclip=${eclip4} grid=y titles="True:Kir1:True:Kir2:True:Kir3:True:Kir4:True:Kir5:True:Kir6:True:Kir7:True:Kir8:True:Kir9:True:Kir10:True:Kir11:True:Kir12" gainpanel=a | Xtpen pixmaps=y &
	Cat axis=3 t1.H t4l.H t1.H t4m.H t1.H t4m.H t1.H t4n.H t1.H t4o.H t1.H t4p.H t1.H t4q.H | Grey color=j newclip=1 bclip=${bclip4} eclip=${eclip4} grid=y titles="True:Kir13:True:Kir14:True:Kir15:True:Kir16:Tru:Kir17:True:Kir18" gainpanel=a | Xtpen pixmaps=y &
	Cat axis=3 t1.H t5a.H t1.H t5b.H t1.H t5c.H t1.H t5d.H t1.H t5e.H t1.H t5f.H t1.H t5g.H t1.H t5h.H t1.H t5i.H t1.H t5j.H t1.H t5k.H t1.H t5l.H | Grey color=j newclip=1 bclip=${bclip4} eclip=${eclip4} grid=y titles="True:Blog1:True:Blog2:True:Blog3:True:Blog4:True:Blog5:True:Blog6:True:Blog7:True:Blog8:True:Blog9:True:Blog10:True:Blog11:True:Blog12" gainpanel=a | Xtpen pixmaps=y &
