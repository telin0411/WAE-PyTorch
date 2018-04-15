# for mmd
python3 main.py --dataroot=data/CelebA --dataset='celebA' --gpu_id=0 --cuda --noise='gaussian' --outf=gan_outputs2/ --mode='gan' --lr=0.0003 --pz_scale=2 --LAMBDA=10 --niter=55 --e_pretrain

# for gan
python3 main.py --dataroot=data/CelebA --dataset='celebA' --gpu_id=1 --cuda --noise='add_noise' --outf=gan_outputs/ --mode='gan' --lr=0.0003 --pz_scale=2 --LAMBDA=10 --niter=55 --input_normalize_sym
