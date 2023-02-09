import subprocess
from itertools import product
import concurrent.futures
from glob import glob
import shutil
import gc
from random import shuffle


arguments = {
    'encoder': ['cnn_no_lat'],
    'enc-actf': ["relu"],
    'enc-dropout': [0.0],
    'enc-layers': [1, 5, 7],
    'decoder': ['transformer', 'entransformer'],
    'dec-actf': ["relu", "leakyrelu"],
    'dec-dropout': [0.0, 0.2, 0.4],
    'dec-layers': [1, 5, 7],
    'dec-iterations': [6, 9],
    'coder': ['cnn_nolat', 'transformer'],
    'coder-actf': ["relu"],
    'coder-dropout': [0.0],
    'coder-layers': [5],
    'lat-redundancy': [0],
    'enc-lr': [0.000001],
    'dec-lr': [0.00001, 0.000001, 0.0001, 0.001],
    'coder-lr': [0.0001],
    'enc-steps': [3],
    'dec-steps': [1, 2, 5],
    'coder-steps': [5],
    'batch-size': [64, 128, 256, 512]
}

#coder-units 64 + redundancy!

def run_param_combination(pcom):
    # Try out the parameter combination here
    name = '_'.join(str(v) for v in pcom.values())
    #if name in already_finished:
    #    print("ignored " + name + " as it is already done.")
    #    return
    print("starting " + name)
    py_command = (
            "./sDNA.py --train --threads 10 --epochs 50 --wdir /home/wintermute/projects/autoturbo_dna/models/tuning_gpu/" + name + " --coder-units "
            + str(32 + pcom["lat-redundancy"]) + " --encoder " + pcom["encoder"] +
            " --enc-actf " + pcom["enc-actf"] + " --enc-dropout " + str(pcom["enc-dropout"]) +
            " --enc-layers " + str(pcom["enc-layers"]) + " --decoder " + pcom["decoder"] +
            " --dec-actf " + pcom["dec-actf"] + " --dec-dropout " + str(pcom["dec-dropout"]) +
            " --dec-layers " + str(pcom["dec-layers"]) + " --dec-iterations " + str(pcom["dec-iterations"]) +
            " --coder " + str(pcom["coder"]) + " --coder-actf " + str(pcom["coder-actf"]) + " --coder-dropout " +
            str(pcom["coder-dropout"]) + " --coder-layers " + str(pcom["coder-layers"]) + " --lat-redundancy " +
            str(pcom["lat-redundancy"]) + " --enc-lr " + str(pcom["enc-lr"]) + " --dec-lr " + str(pcom["dec-lr"])
            + " --coder-lr " + str(pcom["coder-lr"]) + " --enc-steps " + str(pcom["enc-steps"]) + " --dec-steps " + str(
        pcom["dec-steps"])
            + " --coder-steps " + str(pcom["coder-steps"]) + " --batch-size " + str(pcom["batch-size"]))

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
    with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
        combinations = list(product(*values))
        shuffle(combinations)
        for combination in combinations:
            pcom = dict(zip(keys, combination))
            name = '_'.join(str(v) for v in pcom.values())
            if name not in already_finished:
                executor.submit(run_param_combination, pcom)
            gc.collect()

if __name__ == '__main__':
    folders = glob("./models/tuning_random/*")
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

    try_param_combinations(arguments, already_finished)
