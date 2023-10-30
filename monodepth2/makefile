train:
	python train.py --model_name mono_model --png --split eigen_full --data_path /mnt/wext/cv_data/kitti/kitti_raw_data --batch_size 8 --log_frequency 32 --num_workers 8

fine-tune:
	python train.py --model_name finetuned_mono --load_weights_folder ~/tmp/mono_model/models/weights_19 --png

overfit:
	python train.py --no_cuda --exp_disabled --use_modelip_loss --modelip_loss_scale 0.1 --do_overfit --use_line_loss --model_name mono_model --png --split eigen_full --data_path /mnt/wext/cv_data/kitti/kitti_raw_data --batch_size 1 --log_frequency  1 --num_workers 0 --num_epochs 200 --save_frequency 200 --scheduler_step_size 200 --learning_rate 1e-3

exp_name=estimated_puffin_8334
exp_name=big_cuckoo_8361_9
# exp_name=grumpy_board_7469
exp_path=/tmp/artifacts/$(exp_name)
# exp_path=/mnt/wext/msc_studies/monodepth_project/related_work/monodepth2/monodepth2/models/mono_640x192
eval:
	for exp_name in estimated_puffin_8334 big_cuckoo_8361 grumpy_board_7469; do \
		exp_path=/tmp/artifacts/$$exp_name; \
		python evaluate_depth.py --png --eval_mono --load_weights_folder $$exp_path --data_path /mnt/wext/cv_data/kitti/kitti_raw_data --eval_split eigen --eval_out_dir $$exp_path/eval; \
	done

create-gt:
	python export_gt_depth.py --split eigen --data_path /mnt/wext/cv_data/kitti/kitti_raw_data

tb:
	tensorboard --logdir ~/tmp/mono_model/