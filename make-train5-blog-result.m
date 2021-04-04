########################### Result on training set #############################
pt1-blog%:
	python ./python/CTP_main.py predict train5-win1-blog$* --model blog --device cuda:0 --patient_file ${ft1} --patient_id ft1

pd1-blog%:
	python ./python/CTP_main.py predict train5-win1-blog$* --model blog --device cuda:0 --patient_file ${pd1} --patient_id fd1

make-blog-pt1:
	make pt1-blog1 -B
	make pt1-blog2 -B
	make pt1-blog3 -B
	make pt1-blog4 -B
	make pt1-blog5 -B
	make pt1-blog6 -B
	make pt1-blog7 -B
	make pt1-blog8 -B
	make pt1-blog9 -B
	make pt1-blog10 -B
	make pt1-blog11 -B
	make pt1-blog12 -B

make-blog-pd1:
	make pd1-blog1 -B
	make pd1-blog2 -B
	make pd1-blog3 -B
	make pd1-blog4 -B
	make pd1-blog5 -B
	make pd1-blog6 -B
	make pd1-blog7 -B
	make pd1-blog8 -B
	make pd1-blog9 -B
	make pd1-blog10 -B
	make pd1-blog11 -B
	make pd1-blog12 -B

train5-blog-ts%:
	# Training data
	Window3d n3=1 f3=$* < ${ft1}_tmax_m.H > tTrue.H
	Window3d n3=1 f3=$* < models/train5-win1-blog1/train5-win1-blog1_ft1_pred_tmax_sep.H > t1.H
	Window3d n3=1 f3=$* < models/train5-win1-blog2/train5-win1-blog2_ft1_pred_tmax_sep.H > t2.H
	Window3d n3=1 f3=$* < models/train5-win1-blog3/train5-win1-blog3_ft1_pred_tmax_sep.H > t3.H
	Window3d n3=1 f3=$* < models/train5-win1-blog4/train5-win1-blog4_ft1_pred_tmax_sep.H > t4.H
	Window3d n3=1 f3=$* < models/train5-win1-blog5/train5-win1-blog5_ft1_pred_tmax_sep.H > t5.H
	Window3d n3=1 f3=$* < models/train5-win1-blog6/train5-win1-blog6_ft1_pred_tmax_sep.H > t6.H
	Window3d n3=1 f3=$* < models/train5-win1-blog7/train5-win1-blog7_ft1_pred_tmax_sep.H > t7.H
	Window3d n3=1 f3=$* < models/train5-win1-blog8/train5-win1-blog8_ft1_pred_tmax_sep.H > t8.H
	Window3d n3=1 f3=$* < models/train5-win1-blog9/train5-win1-blog9_ft1_pred_tmax_sep.H > t9.H
	Window3d n3=1 f3=$* < models/train5-win1-blog10/train5-win1-blog10_ft1_pred_tmax_sep.H > t10.H
	Window3d n3=1 f3=$* < models/train5-win1-blog11/train5-win1-blog11_ft1_pred_tmax_sep.H > t11.H
	Window3d n3=1 f3=$* < models/train5-win1-blog12/train5-win1-blog12_ft1_pred_tmax_sep.H > t12.H
	Cat axis=3 tTrue.H t1.H tTrue.H t2.H tTrue.H t3.H tTrue.H t4.H tTrue.H t5.H tTrue.H t6.H tTrue.H t7.H tTrue.H t8.H tTrue.H t9.H tTrue.H t10.H tTrue.H t11.H tTrue.H t12.H > temp1.H
	Grey color=j newclip=1 bclip=${bclip_pt1} eclip=${eclip_pt1} grid=y titles="True:1:True:2:True:3:True:4:True:5:True:6:True:7:True:8:True:9:True:10:True:11:True:12" gainpanel=a < temp1.H | Xtpen pixmaps=y &

train5-blog-ds%:
	# Training data
	Window3d n3=1 f3=$* < ${pd1}_tmax_m.H > tTrue.H
	Window3d n3=1 f3=$* < models/train5-win1-blog1/train5-win1-blog1_fd1_pred_tmax_sep.H > t1.H
	Window3d n3=1 f3=$* < models/train5-win1-blog2/train5-win1-blog2_fd1_pred_tmax_sep.H > t2.H
	Window3d n3=1 f3=$* < models/train5-win1-blog3/train5-win1-blog3_fd1_pred_tmax_sep.H > t3.H
	Window3d n3=1 f3=$* < models/train5-win1-blog4/train5-win1-blog4_fd1_pred_tmax_sep.H > t4.H
	Window3d n3=1 f3=$* < models/train5-win1-blog5/train5-win1-blog5_fd1_pred_tmax_sep.H > t5.H
	Window3d n3=1 f3=$* < models/train5-win1-blog6/train5-win1-blog6_fd1_pred_tmax_sep.H > t6.H
	Window3d n3=1 f3=$* < models/train5-win1-blog7/train5-win1-blog7_fd1_pred_tmax_sep.H > t7.H
	Window3d n3=1 f3=$* < models/train5-win1-blog8/train5-win1-blog8_fd1_pred_tmax_sep.H > t8.H
	Window3d n3=1 f3=$* < models/train5-win1-blog9/train5-win1-blog9_fd1_pred_tmax_sep.H > t9.H
	Window3d n3=1 f3=$* < models/train5-win1-blog10/train5-win1-blog10_fd1_pred_tmax_sep.H > t10.H
	Window3d n3=1 f3=$* < models/train5-win1-blog11/train5-win1-blog11_fd1_pred_tmax_sep.H > t11.H
	Window3d n3=1 f3=$* < models/train5-win1-blog12/train5-win1-blog12_fd1_pred_tmax_sep.H > t12.H
	Cat axis=3 tTrue.H t1.H tTrue.H t2.H tTrue.H t3.H tTrue.H t4.H tTrue.H t5.H tTrue.H t6.H tTrue.H t7.H tTrue.H t8.H tTrue.H t9.H tTrue.H t10.H tTrue.H t11.H tTrue.H t12.H > temp1.H
	Grey color=j newclip=1 bclip=${bclip_pt1} eclip=${eclip_pt1} grid=y titles="True:1:True:2:True:3:True:4:True:5:True:6:True:7:True:8:True:9:True:10:True:11:True:12" gainpanel=a < temp1.H | Xtpen pixmaps=y &
