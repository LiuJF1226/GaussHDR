
# exp_mode=1  # training setting of HDR-NeRF
exp_mode=3  # training setting of HDR-GS
data_dir=datasets/HDR-NeRF-syn
exp_dir=logs_exp${exp_mode}/HDR-NeRF-syn
voxel_size=0.001
update_init_factor=16
data_type=synthetic
resolution=2
iterations=30000
gamma=0.5
test_iterations=(30000)
save_iterations=(30000)

## sequential training (one by one)

# for scene in "bathroom" "sponza" "bear" "chair" "diningroom" "sofa" "dog" "desk"; do
#     python train.py --gpu 0 -s ${data_dir}/${scene} -m ${exp_dir}/${scene} -r ${resolution} -d ${data_type} --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --exp_mode ${exp_mode} --iterations ${iterations} --test_iterations ${test_iterations[@]} --save_iterations ${save_iterations[@]} --gamma ${gamma}
# done


## parallel training

python train.py --gpu 0 -s ${data_dir}/bathroom -m ${exp_dir}/bathroom -r ${resolution} -d ${data_type} --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --exp_mode ${exp_mode} --iterations ${iterations} --test_iterations ${test_iterations[@]} --save_iterations ${save_iterations[@]} --gamma ${gamma} &

python train.py --gpu 1 -s ${data_dir}/sponza -m ${exp_dir}/sponza -r ${resolution} -d ${data_type} --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --exp_mode ${exp_mode} --iterations ${iterations} --test_iterations ${test_iterations[@]} --save_iterations ${save_iterations[@]} --gamma ${gamma} &

python train.py --gpu 2 -s ${data_dir}/bear -m ${exp_dir}/bear -r ${resolution} -d ${data_type} --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --exp_mode ${exp_mode} --iterations ${iterations} --test_iterations ${test_iterations[@]} --save_iterations ${save_iterations[@]} --gamma ${gamma} &

python train.py --gpu 3 -s ${data_dir}/chair -m ${exp_dir}/chair -r ${resolution} -d ${data_type} --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --exp_mode ${exp_mode} --iterations ${iterations} --test_iterations ${test_iterations[@]} --save_iterations ${save_iterations[@]} --gamma ${gamma} &

python train.py --gpu 4 -s ${data_dir}/diningroom -m ${exp_dir}/diningroom -r ${resolution} -d ${data_type} --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --exp_mode ${exp_mode} --iterations ${iterations} --test_iterations ${test_iterations[@]} --save_iterations ${save_iterations[@]} --gamma ${gamma}  &

python train.py --gpu 5 -s ${data_dir}/sofa -m ${exp_dir}/sofa -r ${resolution} -d ${data_type} --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --exp_mode ${exp_mode} --iterations ${iterations} --test_iterations ${test_iterations[@]} --save_iterations ${save_iterations[@]} --gamma ${gamma}  &

python train.py --gpu 6 -s ${data_dir}/dog -m ${exp_dir}/dog -r ${resolution} -d ${data_type} --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --exp_mode ${exp_mode} --iterations ${iterations} --test_iterations ${test_iterations[@]} --save_iterations ${save_iterations[@]} --gamma ${gamma}  &

python train.py --gpu 7 -s ${data_dir}/desk -m ${exp_dir}/desk -r ${resolution} -d ${data_type} --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --exp_mode ${exp_mode} --iterations ${iterations} --test_iterations ${test_iterations[@]} --save_iterations ${save_iterations[@]} --gamma ${gamma}  &

wait


