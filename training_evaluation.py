import subprocess
from itertools import product
import concurrent.futures
from glob import glob
import shutil
import gc
from random import shuffle

arguments2 = {
    'encoder': ['vae', 'cnn'],
    'decoder': ['cnn_nolat'],
    'coder': ['resnet'],
    'lat-redundancy': [0],
    'block-length': [8, 16, 32, 64],
    "coder_target": ['encoded_data'] #reconstruction
}

def run_param_combination(pcom):
    # Try out the parameter combination here
    name = '_'.join(str(v) for v in pcom.values())
    #if name in already_finished:
    #    print("ignored " + name + " as it is already done.")
    #    return
    print("starting " + name)
    py_command = ("./sDNA.py --train --threads 5 --epochs 1000 --wdir /PATH/autoturbo_dna/models/final_trainings/"
                  + name + " --coder-units 128 --encoder " + str(pcom["encoder"])
                  + " --enc-dropout 0.2 --enc-layers 7 +  --decoder " + str(pcom["decoder"])
                  + " --dec-layers 7 --dec-iterations 10" + " --lat-redundancy " + str(pcom["lat-redundancy"])
                  + " --enc-lr 0.0001 --dec-lr  0.0001 --enc-steps 20 --dec-steps 30 --coder "
                  + str(pcom["coder"]) + " --coder-layers 7 --coder-lr 0.0001 --coder-steps 20 --block-length "
                  + str(pcom["block-length"]) + " --block-padding " + str(pcom["block-length"]//4)
                  +" --init-weights normal --combined-steps 20 --coder-train-target " + str(pcom["coder_target"])) #continuous-coder constraint-training
    print(py_command)
    process = subprocess.Popen(py_command.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    print(output.decode().split("\n")[-2])
    print(error)
    gc.collect()

def try_param_combinations(param_dict, already_finished):
    keys = param_dict.keys()
    values = param_dict.values()
    #for combination in product(*values):
    #        pcom = dict(zip(keys, combination))
    #        run_param_combination(pcom)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        combinations = list(product(*values))
        shuffle(combinations)
        for combination in combinations:
            pcom = dict(zip(keys, combination))
            name = '_'.join(str(v) for v in pcom.values())
            if name not in already_finished:
                executor.submit(run_param_combination, pcom)
            gc.collect()

if __name__ == '__main__':
    folders = glob("./models/final_trainings/*")
    print(folders)
    already_finished = set()
    for folder in folders:
        try:
            with open(folder + "/summary.txt", 'r') as fp:
                x = len(fp.readlines())
                if x < 51:
                    shutil.rmtree(folder)
                    print("removed folder " + str(folder) + ", as it only had " + str(x) + " lines")
                else:
                    already_finished.add(folder.split("/")[-1])
        except:
            shutil.rmtree(folder)
            print("no summary")

    try_param_combinations(arguments2, already_finished)
