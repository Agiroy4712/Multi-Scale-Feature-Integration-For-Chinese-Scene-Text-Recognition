python main.py --test_data_dir /home/liaohaiqing/dataset/scene/scene_test/ --batch_size 1024 --workers 0 --height 64 --width 256 --voc_type CHINESE_7715 --arch ResNet_ASTER --with_lstm  --logs_dir logs/baseline_aster --real_logs_dir test_result --max_len 200 --evaluate --STN_ON --tps_inputsize 32 64 --tps_outputsize 32 100 --tps_margins 0.05 0.05 --stn_activation none --num_control_points 20 --resume /home/liaohaiqing/Chinese_Scene_Text_Rec_multi_se/ch_aster_multi_se01/model_best.pth.tar --test_save aster_aspp_se_ch01.txt