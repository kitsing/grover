for i in `seq 1000 1000 20000`
do 
for layer in 4
do
for capping in tanh
do
python nce/compute_ppl_ratio.py --inp /checkpoint/kitsing/grover/test_npz/preprocessed_val0000.tfrecord.npz --model-config lm/configs/reverse/${layer}_${capping}.json --dis-ckpt /checkpoint/kitsing/grover-models/april_test/rev/${layer}-${capping}/model.ckpt-${i} --noise-files "/checkpoint/kitsing/grover/unconditional_samples_vanilla/7/unconditioned_1[0-4]*.npz" | tee /checkpoint/kitsing/grover-models/april_test/rev/${layer}-${capping}/${i}.dev.out
done
done
done
