
echo "-------------------------------------------------------------------------------------------------------------"
python test_benchmark_inference.py -p -d /mnt/str/models/llama-7b-4bit-128g
echo "-------------------------------------------------------------------------------------------------------------"
python test_benchmark_inference.py -p -d /mnt/str/models/llama-13b-4bit-128g
echo "-------------------------------------------------------------------------------------------------------------"
python test_benchmark_inference.py -p -d /mnt/str/models/llama-30b-4bit-128g
echo "-------------------------------------------------------------------------------------------------------------"
python test_benchmark_inference.py -p -d /mnt/str/models/llama-30b-4bit-128g-act
echo "-------------------------------------------------------------------------------------------------------------"
python test_benchmark_inference.py -p -d /mnt/str/models/llama-30b-4bit-32g-act-ts -l 1500
echo "-------------------------------------------------------------------------------------------------------------"
python test_benchmark_inference.py -p -d /mnt/str/models/koala-13B-4bit-128g-act
echo "-------------------------------------------------------------------------------------------------------------"
python test_benchmark_inference.py -p -d /mnt/str/models/wizardlm-30b-uncensored-4bit-act-order
echo "-------------------------------------------------------------------------------------------------------------"
