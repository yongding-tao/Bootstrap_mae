nohup sh ./Experiment_Settings/run_Bootstrap_MAE_setting1.sh > ./Experiment_results/Bmae_k_2_pn/screen.log 2>&1 &

sleep 5s

nohup sh ./Experiment_Settings/run_Bootstrap_MAE_setting2.sh > ./Experiment_results/Bmae_k_3_pn/screen.log 2>&1 &

sleep 5s

nohup sh ./Experiment_Settings/run_Bootstrap_MAE_setting4.sh > ./Experiment_results/Bmae_k_5_pn/screen.log 2>&1 &

sleep 5s

nohup sh ./Experiment_Settings/run_Bootstrap_MAE_setting6.sh > ./Experiment_results/Bmae_k_7_pn/screen.log --2>&1 &
