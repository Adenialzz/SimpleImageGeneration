set -e -u

image_dir=n07730033/
out_dir=/media/song/vqgan_outputs_2/
n_epochs=100
device=cuda:1

if [ ! -d $out_dir ]; then
	mkdir -p $out_dir
fi

echo 'VQGAN Training!'
python training_vqgan.py \
	--dataset-path $image_dir \
	--output-path $out_dir \
	--epochs $n_epochs \
	--batch-size 4 \
	--device $device \
	--learning-rate 1e-5
echo 'VQGAN Training Done!'

echo 'Transformer Training!'
python training_transformer.py \
	--dataset-path $image_dir \
	--output-path $out_dir \
	--vqgan-checkpoint-path ${out_dir}/checkpoints/vqgan_epoch_`expr ${n_epochs} - 1`.pt \
	--epochs $n_epochs \
	--batch-size 6 \
	--device $device
echo 'Transformer Training Done!'


echo "Sampling"
python sample_transformer.py \
	--vqgan-checkpoint-path ${out_dir}/checkpoints/vqgan_epoch_`expr ${n_epochs} - 1`.pt \
	--transformer-checkpoint-path ${out_dir}/checkpoints/transformer_`expr ${n_epochs} - 1`.pt \
	--n-samples 50 \
	--device cuda:2 \
	--output-path ${out_dir}/results/
echo "Sampling Done. Results are saved at ${out_dir}/results/sample_results/"
