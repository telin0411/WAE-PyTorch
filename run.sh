python3 main.py --dataroot=data/CelebA --dataset='celebA' --gpu_id=1 --cuda --noise='add_noise' --outf=gan_outputs/ --mode='gan'
#python3 main.py --dataroot=data/CelebA --dataset='celebA' --gpu_id=0 --cuda --noise='add_noise' --outf=mmd_outputs/ --mode='mmd' --kernel='IMQ'
python3 main.py --dataroot=data/CelebA --dataset='celebA' --gpu_id=0 --cuda --noise='add_noise' --outf=mmd_outputs/ --mode='mmd' --kernel='RBF' --lr=1e-4 --LAMBDA=100 --niter=55 --e_pretrain
python3 main.py --dataroot=data/CelebA --dataset='celebA' --gpu_id=1 --cuda --noise='add_noise' --outf=gan_outputs/ --mode='gan' --lr=0.0003 --pz_scale=1 --LAMBDA=10 --niter=55 --e_pretrain
