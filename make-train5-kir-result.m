########################### Result on training set #############################
pt1-kir%:
	python ./python/CTP_main.py predict train5-win1-kir$* --model kiranyaz --device cuda:0 --patient_file ${ft1} --patient_id ft1

pd1-kir%:
	python ./python/CTP_main.py predict train5-win1-kir$* --model kiranyaz --device cuda:0 --patient_file ${pd1} --patient_id fd1

make-kir-pt1:
	# make pt1-kir1 -B
	# make pt1-kir2 -B
	# make pt1-kir3 -B
	# make pt1-kir4 -B
	# make pt1-kir7 -B
	# make pt1-kir8 -B
	# make pt1-kir9 -B
	# make pt1-kir10 -B
	# make pt1-kir11 -B
	# make pt1-kir12 -B
	# make pt1-kir13 -B
	# make pt1-kir14 -B
	# make pt1-kirMae1 -B
	# make pt1-kirMae2 -B
	# make pt1-kirMae3 -B
	make pt1-kirHuber1 -B
	make pt1-kirHuber2 -B
	make pt1-kirHuber3 -B
	make pt1-kirHuber4 -B

make-kir-pd1:
	make pd1-kir1 -B
	make pd1-kir2 -B
	make pd1-kir3 -B
	make pd1-kir4 -B
	make pd1-kir7 -B
	make pd1-kir8 -B
	make pd1-kir9 -B
	make pd1-kir10 -B
	make pd1-kir11 -B
	make pd1-kir12 -B
	make pd1-kir13 -B
	make pd1-kir14 -B
	make pd1-kirMae1 -B
	make pd1-kirMae2 -B
	make pd1-kirMae3 -B
	make pd1-kirHuber1 -B
	make pd1-kirHuber2 -B
	make pd1-kirHuber3 -B
	make pd1-kirHuber4 -B

train5-kir-ts%:
	# Training data
	Window3d n3=1 f3=$* < ${ft1}_tmax_m.H > tTrue.H
	Window3d n3=1 f3=$* < models/train5-win1-kir1/train5-win1-kir1_ft1_pred_tmax_sep.H > t1.H
	Window3d n3=1 f3=$* < models/train5-win1-kir2/train5-win1-kir2_ft1_pred_tmax_sep.H > t2.H
	Window3d n3=1 f3=$* < models/train5-win1-kir3/train5-win1-kir3_ft1_pred_tmax_sep.H > t3.H
	Window3d n3=1 f3=$* < models/train5-win1-kir4/train5-win1-kir4_ft1_pred_tmax_sep.H > t4.H
	Window3d n3=1 f3=$* < models/train5-win1-kir7/train5-win1-kir7_ft1_pred_tmax_sep.H > t7.H
	Window3d n3=1 f3=$* < models/train5-win1-kir8/train5-win1-kir8_ft1_pred_tmax_sep.H > t8.H
	Window3d n3=1 f3=$* < models/train5-win1-kir9/train5-win1-kir9_ft1_pred_tmax_sep.H > t9.H
	Window3d n3=1 f3=$* < models/train5-win1-kir10/train5-win1-kir10_ft1_pred_tmax_sep.H > t10.H
	Window3d n3=1 f3=$* < models/train5-win1-kir11/train5-win1-kir11_ft1_pred_tmax_sep.H > t11.H
	Window3d n3=1 f3=$* < models/train5-win1-kir12/train5-win1-kir12_ft1_pred_tmax_sep.H > t12.H
	Window3d n3=1 f3=$* < models/train5-win1-kir13/train5-win1-kir13_ft1_pred_tmax_sep.H > t13.H
	Window3d n3=1 f3=$* < models/train5-win1-kir14/train5-win1-kir14_ft1_pred_tmax_sep.H > t14.H
	Cat axis=3 tTrue.H t1.H tTrue.H t2.H tTrue.H t3.H tTrue.H t4.H tTrue.H t7.H tTrue.H t8.H tTrue.H t9.H tTrue.H t10.H tTrue.H t11.H tTrue.H t12.H tTrue.H t13.H tTrue.H t14.H > temp1.H
	Grey color=j newclip=1 bclip=${bclip_pt1} eclip=${eclip_pt1} grid=y titles="True:1:True:2:True:3:True:4:True:7:True:8:True:9:True:10:True:11:True:12:True:13:True:14" gainpanel=a < temp1.H | Xtpen pixmaps=y &

train5-kir-ds%:
	# Training data
	Window3d n3=1 f3=$* < ${pd1}_tmax_m.H > tTrue.H
	Window3d n3=1 f3=$* < models/train5-win1-kir1/train5-win1-kir1_fd1_pred_tmax_sep.H > t1.H
	Window3d n3=1 f3=$* < models/train5-win1-kir2/train5-win1-kir2_fd1_pred_tmax_sep.H > t2.H
	Window3d n3=1 f3=$* < models/train5-win1-kir3/train5-win1-kir3_fd1_pred_tmax_sep.H > t3.H
	Window3d n3=1 f3=$* < models/train5-win1-kir4/train5-win1-kir4_fd1_pred_tmax_sep.H > t4.H
	Window3d n3=1 f3=$* < models/train5-win1-kir7/train5-win1-kir7_fd1_pred_tmax_sep.H > t7.H
	Window3d n3=1 f3=$* < models/train5-win1-kir8/train5-win1-kir8_fd1_pred_tmax_sep.H > t8.H
	Window3d n3=1 f3=$* < models/train5-win1-kir9/train5-win1-kir9_fd1_pred_tmax_sep.H > t9.H
	Window3d n3=1 f3=$* < models/train5-win1-kir10/train5-win1-kir10_fd1_pred_tmax_sep.H > t10.H
	Window3d n3=1 f3=$* < models/train5-win1-kir11/train5-win1-kir11_fd1_pred_tmax_sep.H > t11.H
	Window3d n3=1 f3=$* < models/train5-win1-kir12/train5-win1-kir12_fd1_pred_tmax_sep.H > t12.H
	Window3d n3=1 f3=$* < models/train5-win1-kir13/train5-win1-kir13_fd1_pred_tmax_sep.H > t13.H
	Window3d n3=1 f3=$* < models/train5-win1-kir14/train5-win1-kir14_fd1_pred_tmax_sep.H > t14.H
	Cat axis=3 tTrue.H t1.H tTrue.H t2.H tTrue.H t3.H tTrue.H t4.H tTrue.H t7.H tTrue.H t8.H tTrue.H t9.H tTrue.H t10.H tTrue.H t11.H tTrue.H t12.H tTrue.H t13.H tTrue.H t14.H > temp1.H
	Grey color=j newclip=1 bclip=${bclip_pt1} eclip=${eclip_pt1} grid=y titles="True:1:True:2:True:3:True:4:True:7:True:8:True:9:True:10:True:11:True:12:True:13:True:14" gainpanel=a < temp1.H | Xtpen pixmaps=y &
