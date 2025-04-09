
# exp_mode=1  # training setting of HDR-NeRF
exp_mode=3  # training setting of HDR-GS
data_dir=datasets/HDR-Plenoxels-real
exp_dir=logs_exp${exp_mode}/HDR-Plenoxels-real
voxel_size=0.001
update_init_factor=16
data_type=real
resolution=6
gamma=0.2
iterations=30000
test_iterations=(30000)
save_iterations=(30000)


## sequential training (one by one)

# for scene in "coffee" "plant" "character" "desk"; do
#     python train.py --gpu 0 -s ${data_dir}/${scene} -m ${exp_dir}/${scene} -r ${resolution} -d ${data_type} --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --exp_mode ${exp_mode} --iterations ${iterations} --test_iterations ${test_iterations[@]} --save_iterations ${save_iterations[@]} --gamma ${gamma}
# done


## parallel training

python train.py --gpu 0 -s ${data_dir}/coffee -m ${exp_dir}/coffee -r ${resolution} -d ${data_type} --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --exp_mode ${exp_mode} --iterations ${iterations} --test_iterations ${test_iterations[@]} --save_iterations ${save_iterations[@]} --gamma ${gamma} &

python train.py --gpu 1 -s ${data_dir}/plant -m ${exp_dir}/plant -r ${resolution} -d ${data_type} --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --exp_mode ${exp_mode} --iterations ${iterations} --test_iterations ${test_iterations[@]} --save_iterations ${save_iterations[@]} --gamma ${gamma} &

python train.py --gpu 2 -s ${data_dir}/desk -m ${exp_dir}/desk -r ${resolution} -d ${data_type} --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --exp_mode ${exp_mode} --iterations ${iterations} --test_iterations ${test_iterations[@]} --save_iterations ${save_iterations[@]} --gamma ${gamma} &

python train.py --gpu 3 -s ${data_dir}/character -m ${exp_dir}/character -r ${resolution} -d ${data_type} --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --exp_mode ${exp_mode} --iterations ${iterations} --test_iterations ${test_iterations[@]} --save_iterations ${save_iterations[@]} --gamma ${gamma} &

wait