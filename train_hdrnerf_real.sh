
# exp_mode=1  # training setting of HDR-NeRF
exp_mode=3  # training setting of HDR-GS
data_dir=datasets/HDR-NeRF-real
exp_dir=logs_exp${exp_mode}/HDR-NeRF-real
voxel_size=0.001
update_init_factor=16
data_type=real
resolution=4
iterations=30000
gamma=0.2
test_iterations=(24000 30000)
save_iterations=(24000 30000)


## sequential training (one by one)

# python train.py --gpu 0 -s ${data_dir}/box -m ${exp_dir}/box -r ${resolution} -d ${data_type} --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --exp_mode ${exp_mode} --iterations ${iterations} --test_iterations ${test_iterations[@]} --save_iterations ${save_iterations[@]} --gamma ${gamma} 

# python train.py --gpu 0 -s ${data_dir}/computer -m ${exp_dir}/computer -r ${resolution} -d ${data_type} --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --exp_mode ${exp_mode} --iterations ${iterations} --test_iterations ${test_iterations[@]} --save_iterations ${save_iterations[@]} --gamma ${gamma} 

# python train.py --gpu 0 -s ${data_dir}/flower -m ${exp_dir}/flower -r ${resolution} -d ${data_type} --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --exp_mode ${exp_mode} --iterations ${iterations} --test_iterations ${test_iterations[@]} --save_iterations ${save_iterations[@]} --gamma 0.16

# python train.py --gpu 0 -s ${data_dir}/luckycat -m ${exp_dir}/luckycat -r ${resolution} -d ${data_type} --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --exp_mode ${exp_mode} --iterations ${iterations} --test_iterations ${test_iterations[@]} --save_iterations ${save_iterations[@]} --gamma ${gamma} 


## parallel training

python train.py --gpu 0 -s ${data_dir}/box -m ${exp_dir}/box -r ${resolution} -d ${data_type} --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --exp_mode ${exp_mode} --iterations ${iterations} --test_iterations ${test_iterations[@]} --save_iterations ${save_iterations[@]} --gamma ${gamma} &

python train.py --gpu 1 -s ${data_dir}/computer -m ${exp_dir}/computer -r ${resolution} -d ${data_type} --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --exp_mode ${exp_mode} --iterations ${iterations} --test_iterations ${test_iterations[@]} --save_iterations ${save_iterations[@]} --gamma ${gamma} &

python train.py --gpu 2 -s ${data_dir}/flower -m ${exp_dir}/flower -r ${resolution} -d ${data_type} --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --exp_mode ${exp_mode} --iterations ${iterations} --test_iterations ${test_iterations[@]} --save_iterations ${save_iterations[@]} --gamma 0.16 &

python train.py --gpu 3 -s ${data_dir}/luckycat -m ${exp_dir}/luckycat -r ${resolution} -d ${data_type} --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --exp_mode ${exp_mode} --iterations ${iterations} --test_iterations ${test_iterations[@]} --save_iterations ${save_iterations[@]} --gamma ${gamma} &

wait