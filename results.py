import os
import math
import argparse
import numpy as np
from collections import defaultdict


def iterate(path):
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for file in os.listdir(path):
        file_name, ext = file.split("-")
        if ext != "beta.csv":
            continue
        file_path = os.path.join(path, file)
        for line in open(file_path):
            mod, var, val = line.split(",")
            mod = int(mod.strip())
            var = int(var.strip())
            val = float(val.strip())
            data[file_name][mod][var] = val
    return data


def write_set(aggregate):
    metrics = [
        "l1_sp", "l1_mp",
        "l2_sp", "l2_mp",
        "precision_sp", "precision_mp",
        "recall_sp", "recall_mp",
        "accuracy_sp", "accuracy_mp"
    ]
    for category in aggregate:
        f = open(category + "_" + str(len(aggregate[category]["l1_sp"])) + "_table.csv", "w")
        f.write("metrics,mean,min,max,std\n")
        for metric in metrics:
            f.write("%s,%f,%f,%f,%f\n" % (
                metric,
                np.mean(aggregate[category][metric]),
                np.min(aggregate[category][metric]),
                np.max(aggregate[category][metric]),
                np.std(aggregate[category][metric])
            ))
        f.close()


def statistics_set(sp, mp, dp):
    aggregate = defaultdict(lambda: defaultdict(list))

    for file in dp:
        for mod in dp[file]:
            sp_set = sp[file][mod].keys()
            mp_set = mp[file][mod].keys()
            dp_set = dp[file][mod].keys()

            assert sp_set == mp_set
            if sp_set == dp_set:
                is_sign_match = True
                for key in sp_set:
                    if math.copysign(1, mp[file][mod][key]) != math.copysign(1, dp[file][mod][key]):
                        is_sign_match = False
                if is_sign_match:
                    category = "sign_match"
                else:
                    category = "set_match"
            else:
                category = "mismatch"

            precision_sp = len(sp_set & dp_set) / len(sp_set)
            precision_mp = len(mp_set & dp_set) / len(mp_set)

            recall_sp = len(sp_set & dp_set) / len(dp_set)
            recall_mp = len(mp_set & dp_set) / len(dp_set)

            accuracy_sp = len(sp_set & dp_set) / len(sp_set | dp_set)
            accuracy_mp = len(mp_set & dp_set) / len(mp_set | dp_set)

            l1_sp = l2_sp = 0
            l1_mp = l2_mp = 0
            l1_den = l2_den = 0
            for key in (sp_set | dp_set):
                l1_sp += abs(sp[file][mod][key] - dp[file][mod][key])
                l2_sp += (sp[file][mod][key] - dp[file][mod][key]) ** 2
                l1_mp += abs(mp[file][mod][key] - dp[file][mod][key])
                l2_mp += (mp[file][mod][key] - dp[file][mod][key]) ** 2
                l1_den += abs(dp[file][mod][key])
                l2_den += (dp[file][mod][key]) ** 2
            l1_sp /= l1_den
            l2_sp /= l2_den
            l1_mp /= l1_den
            l2_mp /= l2_den

            aggregate[category]["l1_sp"].append(l1_sp)
            aggregate[category]["l1_mp"].append(l1_mp)
            aggregate[category]["l2_sp"].append(l2_sp)
            aggregate[category]["l2_mp"].append(l2_mp)
            aggregate[category]["precision_sp"].append(precision_sp)
            aggregate[category]["precision_mp"].append(precision_mp)
            aggregate[category]["recall_sp"].append(recall_sp)
            aggregate[category]["recall_mp"].append(recall_mp)
            aggregate[category]["accuracy_sp"].append(accuracy_sp)
            aggregate[category]["accuracy_mp"].append(accuracy_mp)
    
    write_set(aggregate)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-sp", type=str, help="Path to SP results.",
                        required=True)
    parser.add_argument("-mp", type=str, help="Path to MP results.",
                        required=True)
    parser.add_argument("-dp", type=str, help="Path to DP results.",
                        required=True)
    args = parser.parse_args()

    sp = iterate(args.sp)
    mp = iterate(args.mp)
    dp = iterate(args.dp)

    statistics_set(sp, mp, dp)


if __name__ == "__main__":
    main()
