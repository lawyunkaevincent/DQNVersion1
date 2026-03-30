Run the code with the following command:

If start fresh:
python main.py --cfg .\RLTrainingMap1\map.sumocfg --episodes 100 --save-every 10 --save-ckpt checkpoints/latest.pkl --best-ckpt checkpoints/best_sarsa.pkl --epsilon-start 1.0 --epsilon-end 0.05 --epsilon-decay 0.99 --reward-file reward.txt

If resume from checkpoint:
python main.py --cfg .\RLTrainingMap1\map.sumocfg --load-ckpt checkpoints/best_sarsa.pkl --episodes 30 --save-every 10 --save-ckpt checkpoints/latest.pkl --best-ckpt checkpoints/best_sarsa.pkl --epsilon-start 1.0 --epsilon-end 0.05 --epsilon-decay 0.99 --reward-file reward.txt

To visualize the best policy:
python visualize_policy.py --cfg .\RLTrainingMap1\map.sumocfg --ckpt checkpoints/best_sarsa.pkl

To run the dispactcher at the DQN folder:
python .\dispatcher.py --cfg ..\SmallTestingMap\map.sumocfg   

To generate the request: 
python .\request_chain_generator.py --report D:\FYP\FYP-DRT\SmallTestingMap\connectivity_report.json --taxi D:\FYP\FYP-DRT\SmallTestingMap\map.rou.xml --output D:\FYP\FYP-DRT\SmallTestingMap\persontrips_scale.rou.xml --num-requests 200 --depart-step 25 75 200 --max-random-deviation-pct 10

To collect data for training:
python .\collect_imitation_dataset.py --cfg D:\6Sumo\DQNImitation\SmallTestingMap\map.sumocfg

To train the DQN model:
python train_dqn.py --cfg D:\FYP\DQNVersion1\SmallTestingMap\map.sumocfg --imitation-model-dir artifacts\imitation_model --output-dir artifacts\dqn_model --episodes 40 --batch-size 64 --replay-size 20000 --warmup-transitions 100 --gamma 0.99 --lr 0.0001 --tau 0.01 --epsilon-start 0.10 --epsilon-end 0.02

python train_dqn.py --cfg D:\FYP\DQNVersion1\SmallTestingMap\map.sumocfg --imitation-model-dir artifacts/imitation_model --output-dir artifacts/dqn_model --episodes 100 --warmup-episodes 3 --train-every 4 --batch-size 128 --replay-size 50000 --gamma 0.95 --lr 1e-4 --lr-min 1e-5 --tau 0.005 --epsilon-start 0.10 --epsilon-end 0.01 --eval-every 3

To test the DQN model: 
python run_dqn_policy.py --cfg D:\path\to\map.sumocfg --model-dir artifacts\dqn_model

To test the imitation model:
with fallback: 
python run_imitation_policy.py --cfg D:\6Sumo\DQNImitation\SmallTestingMap\map.sumocfg  --model-dir artifacts/imitation_model --heuristic-fallback-gap 0.25
without fallback:
python run_imitation_policy.py --cfg D:\6Sumo\DQNImitation\SmallTestingMap\map.sumocfg  --model-dir artifacts/imitation_model --heuristic-fallback-gap -1


how the top K (K is how many) candidates is chosen, if the list for t_1 is long and for t_0 is short then most likely there
will be more t_1 candidate then t_0 candidate, how to solve this?