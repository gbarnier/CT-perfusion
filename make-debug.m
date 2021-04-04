################################################################################
################################### Training ###################################
################################################################################
debug-kir1:
	rm -rf models/debug-kir1
	python ./python/CTP_main.py train debug-kir1 --model kiranyaz --n_epochs 20 --lr=0.5e-4 --device cuda:0 --train_file_list ${train5_file} --dev_file_list ${dev5_file} --batch_size 64

debug-kir2:
	rm -rf models/debug-kir2
	python ./python/CTP_main.py train debug-kir2 --model kiranyaz --n_epochs 20 --lr=0.5e-4 --device cuda:1 --train_file_list ${train5_file} --dev_file_list ${dev5_file} --batch_size 128

debug-kir3:
	rm -rf models/debug-kir3
	python ./python/CTP_main.py train debug-kir3 --model kiranyaz --n_epochs 20 --lr=0.5e-4 --device cuda:2 --train_file_list ${train5_file} --dev_file_list ${dev5_file} --batch_size 512

debug-kir4:
	rm -rf models/debug-kir4
	python ./python/CTP_main.py train debug-kir4 --model kiranyaz --n_epochs 20 --lr=0.5e-4 --device cuda:3 --train_file_list ${train5_file} --dev_file_list ${dev5_file} --batch_size 1024

debug-kir5:
	rm -rf models/debug-kir5
	python ./python/CTP_main.py train debug-kir5 --model gui --n_epochs 5 --lr=0.5e-4 --device cuda:3 --train_file_list ${train5_file} --dev_file_list ${dev5_file} --batch_size 1024

################################################################################
################################### Testing ####################################
################################################################################
debug-test-kir1:
	python ./python/CTP_main.py test debug-kir1 --model kiranyaz --device cuda:0 --train_file_list ${train5_file} --dev_file_list ${dev5_file} --test_file_list ${test5_file}

debug-test-kir2:
	python ./python/CTP_main.py test debug-kir2 --model kiranyaz --device cuda:1 --train_file_list ${train5_file} --dev_file_list ${dev5_file} --test_file_list ${test5_file}

debug-test-kir3:
	python ./python/CTP_main.py test debug-kir1 --model kiranyaz --device cuda:2 --train_file_list ${train5_file} --dev_file_list ${dev5_file} --test_file_list ${test5_file}

debug-test-kir4:
	python ./python/CTP_main.py test debug-kir4 --model kiranyaz --device cuda:3 --train_file_list ${train5_file} --dev_file_list ${dev5_file} --test_file_list ${test5_file}

################################################################################
################################### Predicting #################################
################################################################################
debug_test_file=dat/S00243-train5-win1.h5

debug-pred-kir1:
	python ./python/CTP_main.py predict debug-kir1 --model kiranyaz --device cuda:0 --patient_file ${debug_test_file} --patient_id debugTest

debug-pred-kir2:
	python ./python/CTP_main.py predict debug-kir2 --model kiranyaz --device cuda:0 --patient_file ${debug_test_file} --patient_id debugTest

debug-pred-kir3:
	python ./python/CTP_main.py predict debug-kir3 --model kiranyaz --device cuda:0 --patient_file ${debug_test_file} --patient_id debugTest

debug-pred-kir4:
	python ./python/CTP_main.py predict debug-kir4 --model kiranyaz --device cuda:0 --patient_file ${debug_test_file} --patient_id debugTest

dispDebug-s%:
	Window3d n3=1 f3=$* < ${debug_test_file}_tmax_m.H > t0.H
	Window3d n3=1 f3=$* < models/debug-kir1/debug-kir1_debugTest_pred_tmax_sep.H > t1.H
	Window3d n3=1 f3=$* < models/debug-kir2/debug-kir2_debugTest_pred_tmax_sep.H > t2.H
	Window3d n3=1 f3=$* < models/debug-kir3/debug-kir3_debugTest_pred_tmax_sep.H > t3.H
	Window3d n3=1 f3=$* < models/debug-kir4/debug-kir4_debugTest_pred_tmax_sep.H > t4.H
	Cat axis=3 t0.H t1.H t2.H t3.H t4.H | Grey color=j newclip=1 bclip=${bclip5} eclip=${eclip5} grid=y titles="True:1:2:3:4" gainpanel=a | Xtpen pixmaps=y &
