# Train the intention encoder-decoder
python train_MemoNet.py --info try1 --gpu 2 --mode intention

# Train the addressor, stage: warm up
python train_MemoNet.py --info try1 --gpu 2 --mode addressor_warm --model_ae ./training/training_ae/model_encdec

# Train the addressor, stage: finetune
python train_MemoNet.py --info try1 --gpu 2 --mode addressor --model_ae ./training/training_selector/model_selector_warm_up

# Train the trajectory encoder-decoder
python train_MemoNet.py --info try1 --gpu 2 --mode trajectory --model_ae ./training/training_trajectory/model_encdec_trajectory