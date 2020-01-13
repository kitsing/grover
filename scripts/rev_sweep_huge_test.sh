layer=${1}
capping=${2}
i=${3}
chunk_size=${4}
python nce/compute_ppl_ratio.py --chunk-size ${chunk_size}  --inp /checkpoint/kitsing/grover/test_npz/preprocessed_test0000.tfrecord.npz --model-config lm/configs/reverse/${layer}_${capping}.json --dis-ckpt /checkpoint/kitsing/grover-models/april_test/rev/${layer}-${capping}/model.ckpt-${i} --noise-files "/checkpoint/kitsing/grover/unconditional_samples_vanilla/7/unconditioned_[0-1]*.npz" | tee /checkpoint/kitsing/grover-models/april_test/rev/${layer}-${capping}/${i}.test.out
