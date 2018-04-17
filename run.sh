# for mmd
python3 main.py --dataroot=data/celebA --dataset='celebA' --gpu_id=0 --cuda --noise='gaussian' --outf=mmd_outputs/ --mode='mmd' --lr=0.0003 --pz_scale=2 --LAMBDA=100 --niter=55 --kernel='IMQ' --e_pretrain

# for gan
python3 main.py --dataroot=data/celebA --dataset='celebA' --gpu_id=0 --cuda --noise='add_noise' --outf=gan_outputs/ --mode='gan' --lr=0.0003 --pz_scale=2 --LAMBDA=10 --niter=55 --input_normalize_sym
