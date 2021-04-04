########################### Result on training set #############################
ft1=dat/S00243-train5-win1.h5
pd1=dat/S00288-train5-win1.h5

pt1-fc%:
	python ./python/CTP_main.py predict train5-win1-fc$* --model fc6 --device cuda:0 --patient_file ${ft1} --patient_id ft1

pd1-fc%:
	python ./python/CTP_main.py predict train5-win1-fc$* --model fc6 --device cuda:0 --patient_file ${pd1} --patient_id fd1

make-pt1:
	# Fully connected
	make pt1-fc1 -B
	make pt1-fc2 -B
	make pt1-fc3 -B
	make pt1-fc4 -B
	make pt1-fc5 -B
	make pt1-fc6 -B
	make pt1-fc7 -B
	make pt1-fc8 -B
	make pt1-fc9 -B
	make pt1-fc10 -B
	make pt1-fc11 -B
	make pt1-fc12 -B
	make pt1-fc13 -B
	make pt1-fc14 -B
	make pt1-fc15 -B
	make pt1-fc16 -B
	make pt1-fcHuber1 -B
	make pt1-fcHuber2 -B
	make pt1-fcHuber3 -B
	make pt1-fcHuber4 -B
	make pt1-fcHuber5 -B
	make pt1-fcMae1 -B
	make pt1-fcMae2 -B
	make pt1-fcMae3 -B
	make pt1-fcMae4 -B

make-pd1:
	# Fully connected
	make pd1-fc1 -B
	make pd1-fc2 -B
	make pd1-fc3 -B
	make pd1-fc4 -B
	make pd1-fc5 -B
	make pd1-fc6 -B
	make pd1-fc7 -B
	make pd1-fc8 -B
	make pd1-fc9 -B
	make pd1-fc10 -B
	make pd1-fc11 -B
	make pd1-fc12 -B
	make pd1-fc13 -B
	make pd1-fc14 -B
	make pd1-fc15 -B
	make pd1-fc16 -B
	make pd1-fcHuber1 -B
	make pd1-fcHuber2 -B
	make pd1-fcHuber3 -B
	make pd1-fcHuber4 -B
	make pd1-fcHuber5 -B
	make pd1-fcMae1 -B
	make pd1-fcMae2 -B
	make pd1-fcMae3 -B
	make pd1-fcMae4 -B


eclip_pt1=300
bclip_pt1=0
train5-ts%:
	# Training data
	Window3d n3=1 f3=$* < ${ft1}_tmax_m.H > tTrue.H
	Window3d n3=1 f3=$* < models/train5-win1-fc1/train5-win1-fc1_ft1_pred_tmax_sep.H > t1.H
	Window3d n3=1 f3=$* < models/train5-win1-fc2/train5-win1-fc2_ft1_pred_tmax_sep.H > t2.H
	Window3d n3=1 f3=$* < models/train5-win1-fc3/train5-win1-fc3_ft1_pred_tmax_sep.H > t3.H
	Window3d n3=1 f3=$* < models/train5-win1-fc4/train5-win1-fc4_ft1_pred_tmax_sep.H > t4.H
	Window3d n3=1 f3=$* < models/train5-win1-fc5/train5-win1-fc5_ft1_pred_tmax_sep.H > t5.H
	Window3d n3=1 f3=$* < models/train5-win1-fc6/train5-win1-fc6_ft1_pred_tmax_sep.H > t6.H
	Window3d n3=1 f3=$* < models/train5-win1-fc7/train5-win1-fc7_ft1_pred_tmax_sep.H > t7.H
	Window3d n3=1 f3=$* < models/train5-win1-fc8/train5-win1-fc8_ft1_pred_tmax_sep.H > t8.H
	Window3d n3=1 f3=$* < models/train5-win1-fc9/train5-win1-fc9_ft1_pred_tmax_sep.H > t9.H
	Window3d n3=1 f3=$* < models/train5-win1-fc10/train5-win1-fc10_ft1_pred_tmax_sep.H > t10.H
	Window3d n3=1 f3=$* < models/train5-win1-fc11/train5-win1-fc11_ft1_pred_tmax_sep.H > t11.H
	Window3d n3=1 f3=$* < models/train5-win1-fc12/train5-win1-fc12_ft1_pred_tmax_sep.H > t12.H
	Window3d n3=1 f3=$* < models/train5-win1-fc13/train5-win1-fc13_ft1_pred_tmax_sep.H > t13.H
	Window3d n3=1 f3=$* < models/train5-win1-fc14/train5-win1-fc14_ft1_pred_tmax_sep.H > t14.H
	Window3d n3=1 f3=$* < models/train5-win1-fc15/train5-win1-fc15_ft1_pred_tmax_sep.H > t15.H
	Window3d n3=1 f3=$* < models/train5-win1-fc16/train5-win1-fc16_ft1_pred_tmax_sep.H > t16.H
	Window3d n3=1 f3=$* < models/train5-win1-fcHuber1/train5-win1-fcHuber1_ft1_pred_tmax_sep.H > h1.H
	Window3d n3=1 f3=$* < models/train5-win1-fcHuber2/train5-win1-fcHuber2_ft1_pred_tmax_sep.H > h2.H
	Window3d n3=1 f3=$* < models/train5-win1-fcHuber3/train5-win1-fcHuber3_ft1_pred_tmax_sep.H > h3.H
	Window3d n3=1 f3=$* < models/train5-win1-fcHuber4/train5-win1-fcHuber4_ft1_pred_tmax_sep.H > h4.H
	Window3d n3=1 f3=$* < models/train5-win1-fcHuber5/train5-win1-fcHuber5_ft1_pred_tmax_sep.H > h5.H
	Window3d n3=1 f3=$* < models/train5-win1-fcMae1/train5-win1-fcMae1_ft1_pred_tmax_sep.H > m1.H
	Window3d n3=1 f3=$* < models/train5-win1-fcMae2/train5-win1-fcMae2_ft1_pred_tmax_sep.H > m2.H
	Window3d n3=1 f3=$* < models/train5-win1-fcMae3/train5-win1-fcMae3_ft1_pred_tmax_sep.H > m3.H
	Window3d n3=1 f3=$* < models/train5-win1-fcMae4/train5-win1-fcMae4_ft1_pred_tmax_sep.H > m4.H
	Cat axis=3 tTrue.H t1.H tTrue.H t2.H tTrue.H t3.H tTrue.H t4.H tTrue.H t5.H tTrue.H t6.H tTrue.H t7.H tTrue.H t8.H tTrue.H t9.H tTrue.H t10.H tTrue.H t11.H tTrue.H t12.H tTrue.H t13.H tTrue.H t14.H tTrue.H t15.H tTrue.H t16.H > temp1.H
	Cat axis=3 tTrue.H h1.H tTrue.H h2.H tTrue.H h3.H tTrue.H h4.H tTrue.H h5.H > temp2.H
	Cat axis=3 tTrue.H m1.H tTrue.H m2.H tTrue.H m3.H tTrue.H m4.H tTrue.H h5.H > temp3.H
	Grey color=j newclip=1 bclip=${bclip_pt1} eclip=${eclip_pt1} grid=y titles="True:1:True:2:True:3:True:4:True:5:True:6:True:7:True:8:True:9:True:10:True:11:True:12:True:13:True:14:True:15:True:16" gainpanel=a < temp1.H | Xtpen pixmaps=y &
	# Grey color=j newclip=1 bclip=${bclip_pt1} eclip=${eclip_pt1} grid=y titles="True:h1:True:h2:True:h3:True:h4:True:h5" gainpanel=a < temp2.H | Xtpen pixmaps=y &
	# Grey color=j newclip=1 bclip=${bclip_pt1} eclip=${eclip_pt1} grid=y titles="True:m1:True:m2:True:m3:True:m4:True:m5" gainpanel=a < temp3.H | Xtpen pixmaps=y &

train5-ds%:
	# Training data
	Window3d n3=1 f3=$* < ${pd1}_tmax_m.H > tTrue.H
	Window3d n3=1 f3=$* < models/train5-win1-fc1/train5-win1-fc1_fd1_pred_tmax_sep.H > t1.H
	Window3d n3=1 f3=$* < models/train5-win1-fc2/train5-win1-fc2_fd1_pred_tmax_sep.H > t2.H
	Window3d n3=1 f3=$* < models/train5-win1-fc3/train5-win1-fc3_fd1_pred_tmax_sep.H > t3.H
	Window3d n3=1 f3=$* < models/train5-win1-fc4/train5-win1-fc4_fd1_pred_tmax_sep.H > t4.H
	Window3d n3=1 f3=$* < models/train5-win1-fc5/train5-win1-fc5_fd1_pred_tmax_sep.H > t5.H
	Window3d n3=1 f3=$* < models/train5-win1-fc6/train5-win1-fc6_fd1_pred_tmax_sep.H > t6.H
	Window3d n3=1 f3=$* < models/train5-win1-fc7/train5-win1-fc7_fd1_pred_tmax_sep.H > t7.H
	Window3d n3=1 f3=$* < models/train5-win1-fc8/train5-win1-fc8_fd1_pred_tmax_sep.H > t8.H
	Window3d n3=1 f3=$* < models/train5-win1-fc9/train5-win1-fc9_fd1_pred_tmax_sep.H > t9.H
	Window3d n3=1 f3=$* < models/train5-win1-fc10/train5-win1-fc10_fd1_pred_tmax_sep.H > t10.H
	Window3d n3=1 f3=$* < models/train5-win1-fc11/train5-win1-fc11_fd1_pred_tmax_sep.H > t11.H
	Window3d n3=1 f3=$* < models/train5-win1-fc12/train5-win1-fc12_fd1_pred_tmax_sep.H > t12.H
	Window3d n3=1 f3=$* < models/train5-win1-fc13/train5-win1-fc13_fd1_pred_tmax_sep.H > t13.H
	Window3d n3=1 f3=$* < models/train5-win1-fc14/train5-win1-fc14_fd1_pred_tmax_sep.H > t14.H
	Window3d n3=1 f3=$* < models/train5-win1-fc15/train5-win1-fc15_fd1_pred_tmax_sep.H > t15.H
	Window3d n3=1 f3=$* < models/train5-win1-fc16/train5-win1-fc16_fd1_pred_tmax_sep.H > t16.H
	Window3d n3=1 f3=$* < models/train5-win1-fcHuber1/train5-win1-fcHuber1_fd1_pred_tmax_sep.H > h1.H
	Window3d n3=1 f3=$* < models/train5-win1-fcHuber2/train5-win1-fcHuber2_fd1_pred_tmax_sep.H > h2.H
	Window3d n3=1 f3=$* < models/train5-win1-fcHuber3/train5-win1-fcHuber3_fd1_pred_tmax_sep.H > h3.H
	Window3d n3=1 f3=$* < models/train5-win1-fcHuber4/train5-win1-fcHuber4_fd1_pred_tmax_sep.H > h4.H
	Window3d n3=1 f3=$* < models/train5-win1-fcHuber5/train5-win1-fcHuber5_fd1_pred_tmax_sep.H > h5.H
	Window3d n3=1 f3=$* < models/train5-win1-fcMae1/train5-win1-fcMae1_fd1_pred_tmax_sep.H > m1.H
	Window3d n3=1 f3=$* < models/train5-win1-fcMae2/train5-win1-fcMae2_fd1_pred_tmax_sep.H > m2.H
	Window3d n3=1 f3=$* < models/train5-win1-fcMae3/train5-win1-fcMae3_fd1_pred_tmax_sep.H > m3.H
	Window3d n3=1 f3=$* < models/train5-win1-fcMae4/train5-win1-fcMae4_fd1_pred_tmax_sep.H > m4.H
	Cat axis=3 tTrue.H t1.H tTrue.H t2.H tTrue.H t3.H tTrue.H t4.H tTrue.H t5.H tTrue.H t6.H tTrue.H t7.H tTrue.H t8.H tTrue.H t9.H tTrue.H t10.H tTrue.H t11.H tTrue.H t12.H tTrue.H t13.H tTrue.H t14.H tTrue.H t15.H tTrue.H t16.H > temp1.H
	# Cat axis=3 tTrue.H t3.H tTrue.H t7.H tTrue.H t8.H tTrue.H t14.H > temp1.H
	# Cat axis=3 tTrue.H h1.H tTrue.H h2.H tTrue.H h3.H tTrue.H h4.H tTrue.H h5.H > temp2.H
	# Cat axis=3 tTrue.H m1.H tTrue.H m2.H tTrue.H m3.H tTrue.H m4.H tTrue.H h5.H > temp3.H
	Grey color=j newclip=1 bclip=${bclip_pt1} eclip=${eclip_pt1} grid=y titles="True:1:True:2:True:3:True:4:True:5:True:6:True:7:True:8:True:9:True:10:True:11:True:12:True:13:True:14:True:15:True:16" gainpanel=a < temp1.H | Xtpen pixmaps=y &
	# Grey color=j newclip=1 bclip=${bclip_pt1} eclip=${eclip_pt1} grid=y titles="True:3:True:7:True:8:True:14:" gainpanel=a < temp1.H | Xtpen pixmaps=y &
	# Grey color=j newclip=1 bclip=${bclip_pt1} eclip=${eclip_pt1} grid=y titles="True:h1:True:h2:True:h3:True:h4:True:h5" gainpanel=a < temp2.H | Xtpen pixmaps=y &
	# Grey color=j newclip=1 bclip=${bclip_pt1} eclip=${eclip_pt1} grid=y titles="True:m1:True:m2:True:m3:True:m4:True:m5" gainpanel=a < temp3.H | Xtpen pixmaps=y &
