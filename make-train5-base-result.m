########################### Result on training set #############################
pt1-base%:
	python ./python/CTP_main.py predict train5-win1-base$* --model baseline --device cuda:0 --patient_file ${ft1} --patient_id ft1 --baseline_n_hidden 100

pd1-base%:
	python ./python/CTP_main.py predict train5-win1-base$* --model baseline --device cuda:0 --patient_file ${pd1} --patient_id fd1 --baseline_n_hidden 100

make-base-pt1:
	make pt1-base1 -B
	make pt1-base2 -B
	make pt1-base3 -B
	make pt1-base4 -B

make-base-pd1:
	make pd1-base1 -B
	make pd1-base2 -B
	make pd1-base3 -B
	make pd1-base4 -B

train5-base-ts%:
	# Training data
	Window3d n3=1 f3=$* < ${ft1}_tmax_m.H > tTrue.H
	Window3d n3=1 f3=$* < models/train5-win1-base1/train5-win1-base1_ft1_pred_tmax_sep.H > t1.H
	Window3d n3=1 f3=$* < models/train5-win1-base2/train5-win1-base2_ft1_pred_tmax_sep.H > t2.H
	Window3d n3=1 f3=$* < models/train5-win1-base3/train5-win1-base3_ft1_pred_tmax_sep.H > t3.H
	Window3d n3=1 f3=$* < models/train5-win1-base4/train5-win1-base4_ft1_pred_tmax_sep.H > t4.H
	Cat axis=3 tTrue.H t1.H tTrue.H t2.H tTrue.H t3.H tTrue.H t4.H > temp1.H
	Grey color=j newclip=1 bclip=${bclip_pt1} eclip=${eclip_pt1} grid=y titles="True:1:True:2:True:3:True:4" gainpanel=a < temp1.H | Xtpen pixmaps=y &

train5-base-ds%:
	# Training data
	Window3d n3=1 f3=$* < ${pd1}_tmax_m.H > tTrue.H
	Window3d n3=1 f3=$* < models/train5-win1-base1/train5-win1-base1_fd1_pred_tmax_sep.H > t1.H
	Window3d n3=1 f3=$* < models/train5-win1-base2/train5-win1-base2_fd1_pred_tmax_sep.H > t2.H
	Window3d n3=1 f3=$* < models/train5-win1-base3/train5-win1-base3_fd1_pred_tmax_sep.H > t3.H
	Window3d n3=1 f3=$* < models/train5-win1-base4/train5-win1-base4_fd1_pred_tmax_sep.H > t4.H
	Cat axis=3 tTrue.H t1.H tTrue.H t2.H tTrue.H t3.H tTrue.H t4.H > temp1.H
	Grey color=j newclip=1 bclip=${bclip_pt1} eclip=${eclip_pt1} grid=y titles="True:1:True:2:True:3:True:4" gainpanel=a < temp1.H | Xtpen pixmaps=y &
