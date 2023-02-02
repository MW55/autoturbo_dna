import subprocess
from itertools import product

arguments = {
    'encoder': ['cnn', 'cnn_no_lat'],
    'enc-actf': ["relu", "leakyrelu"],
    'enc-dropout': [0.0, 0.2, 0.4],
    'enc-layers': [1, 5, 7],
    'decoder': ['cnn', 'cnn_no_lat'],
    'dec-actf': ["relu", "leakyrelu"],
    'dec-dropout': [0.0, 0.2, 0.4],
    'dec-layers': [1, 5, 7],
    'dec-iterations': [6, 9],
    'coder': ['cnn'],
    'coder-actf': ["relu", "leakyrelu"],
    'coder-dropout': [0.0, 0.2, 0.4],
    'coder-layers': [1, 5, 7],
    'lat-redundancy': [0, 2, 4],
    'enc-lr': [0.00001, 0.000001, 0.0001],
    'dec-lr': [0.00001, 0.000001, 0.0001],
    'coder-lr': [0.001, 0.0001, 0.0001],
    'enc-steps': [1, 3, 5],
    'dec-steps': [1, 2, 5],
    'coder-steps': [2, 5, 7],
    'batch-size': [128, 256, 512]
}

#coder-units 64 + redundancy!


def try_param_combinations(param_dict):
    c = 0
    keys = param_dict.keys()
    values = param_dict.values()
    for combination in product(*values):
        pcom = dict(zip(keys, combination))
        name = '_'.join(str(v) for v in pcom.values())
        py_command = ("./sDNA.py --train --gpu --epochs 50 --wdir /home/wintermute/projects/autoturbo_dna/models/tuning/" + name + " --coder-units "
                      + str(64 + pcom["lat-redundancy"]) + " --encoder " + pcom["encoder"] +
                      " --enc-actf " + pcom["enc-actf"] + " --enc-dropout " + str(pcom["enc-dropout"]) +
                      " --enc-layers " + str(pcom["enc-layers"]) + " --decoder " + pcom["decoder"] +
                      " --dec-actf " + pcom["dec-actf"] + " --dec-dropout " + str(pcom["dec-dropout"]) +
                      " --dec-layers " + str(pcom["dec-layers"]) + " --dec-iterations " + str(pcom["dec-iterations"]) +
                      " --coder " + str(pcom["coder"]) + " --coder-actf " + str(pcom["coder-actf"]) + " --coder-dropout " +
                      str(pcom["coder-dropout"]) + " --coder-layers " + str(pcom["coder-layers"]) + " --lat-redundancy " +
                      str(pcom["lat-redundancy"]) + " --enc-lr " + str(pcom["enc-lr"]) + " --dec-lr " + str(pcom["dec-lr"])
                      + " --coder-lr " + str(pcom["coder-lr"]) + " --enc-steps " + str(pcom["enc-steps"]) + " --dec-steps " + str(pcom["dec-steps"])
                      + " --coder-steps " + str(pcom["coder-steps"]) + " --batch-size " + str(pcom["batch-size"]))
        print(py_command)
        print(c)
        process = subprocess.Popen(py_command.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        print(output.decode().split("\n")[-2])
        print(error)
        # Try out the parameter combination here
        #print(param_combination)
        c+=1

if __name__ == '__main__':
    try_param_combinations(arguments)