import os
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


def write_set(precision, recall, accuracy, l1, l2, fname):
    mean = []
    mean.extend(list(np.mean(precision, axis=0)))
    mean.extend(list(np.mean(recall, axis=0)))
    mean.extend(list(np.mean(accuracy, axis=0)))
    mean.extend(list(np.mean(l1, axis=0)))
    mean.extend(list(np.mean(l2, axis=0)))

    std = []
    std.extend(list(np.std(precision, axis=0)))
    std.extend(list(np.std(recall, axis=0)))
    std.extend(list(np.std(accuracy, axis=0)))
    std.extend(list(np.std(l1, axis=0)))
    std.extend(list(np.std(l2, axis=0)))

    min_ = []
    min_.extend(list(np.min(precision, axis=0)))
    min_.extend(list(np.min(recall, axis=0)))
    min_.extend(list(np.min(accuracy, axis=0)))
    min_.extend(list(np.min(l1, axis=0)))
    min_.extend(list(np.min(l2, axis=0)))

    max_ = []
    max_.extend(list(np.max(precision, axis=0)))
    max_.extend(list(np.max(recall, axis=0)))
    max_.extend(list(np.max(accuracy, axis=0)))
    max_.extend(list(np.max(l1, axis=0)))
    max_.extend(list(np.max(l2, axis=0)))

    f = open(fname + "-" + str(precision.shape[0]), "w")
    f.write("PSPMean, PMPMean, RSPMean, RMPMean, ASPMean, AMPMean, " +
            "1SPMean, 1MPMean, 2SPMean, 2MPMean, " +
            "PSPStd, PMPStd, RSPStd, RMPStd, ASPStd, AMPStd, " +
            "1SPStd, 1MPStd, 2SPStd, 2MPStd, " +
            "PSPMin, PMPMin, RSPMin, RMPMin, ASPMin, AMPMin, " +
            "1SPMin, 1MPMin, 2SPMin, 2MPMin, " +
            "PSPMax, PMPMax, RSPMax, RMPMax, ASPMax, AMPMax, " +
            "1SPMax, 1MPMax, 2SPMax, 2MPMax\n")
    f.write(str(mean[0]) + ", " + str(mean[1]) + ", " +
            str(mean[2]) + ", " + str(mean[3]) + ", " +
            str(mean[4]) + ", " + str(mean[5]) + ", " +
            str(mean[6]) + ", " + str(mean[7]) + ", " +
            str(mean[8]) + ", " + str(mean[9]) + ", " +

            str(std[0]) + ", " + str(std[1]) + ", " +
            str(std[2]) + ", " + str(std[3]) + ", " +
            str(std[4]) + ", " + str(std[5]) + ", " +
            str(std[6]) + ", " + str(std[7]) + ", " +
            str(std[8]) + ", " + str(std[9]) + ", " +

            str(min_[0]) + ", " + str(min_[1]) + ", " +
            str(min_[2]) + ", " + str(min_[3]) + ", " +
            str(min_[4]) + ", " + str(min_[5]) + ", " +
            str(min_[6]) + ", " + str(min_[7]) + ", " +
            str(min_[8]) + ", " + str(min_[9]) + ", " +

            str(max_[0]) + ", " + str(max_[1]) + ", " +
            str(max_[2]) + ", " + str(max_[3]) + ", " +
            str(max_[4]) + ", " + str(max_[5]) + ", " +
            str(max_[6]) + ", " + str(max_[7]) + ", " +
            str(max_[8]) + ", " + str(max_[9]) + "\n")
    f.close()


def statistics_set(sp, mp, dp):
    precision = []
    recall = []
    accuracy = []
    l1 = []
    l2 = []

    eprecision = []
    erecall = []
    eaccuracy = []
    el1 = []
    el2 = []

    def adder(l1, l2, data, flag):
        if flag:
            l1.append(data)
        else:
            l2.append(data)

    for file in dp:
        for mod in dp[file]:
            sp_set = sp[file][mod].keys()
            mp_set = mp[file][mod].keys()
            dp_set = dp[file][mod].keys()
            e = False
            if sp_set == dp_set:
                e = True

            precision_sp = len(sp_set & dp_set) / len(sp_set)
            precision_mp = len(mp_set & dp_set) / len(mp_set)
            adder(eprecision, precision, [precision_sp, precision_mp], e)

            recall_sp = len(sp_set & dp_set) / len(dp_set)
            recall_mp = len(mp_set & dp_set) / len(dp_set)
            adder(erecall, recall, [recall_sp, recall_mp], e)

            accuracy_sp = len(sp_set & dp_set) / len(sp_set | dp_set)
            accuracy_mp = len(mp_set & dp_set) / len(mp_set | dp_set)
            adder(eaccuracy, accuracy, [accuracy_sp, accuracy_mp], e)

            l1_sp = 0
            l1_sp_den = 0
            l2_sp = 0
            l2_sp_den = 0
            for key in (sp_set | dp_set):
                l1_sp += abs(sp[file][mod][key] - dp[file][mod][key])
                l1_sp_den += abs(dp[file][mod][key])
                l2_sp += (sp[file][mod][key] - dp[file][mod][key]) ** 2
                l2_sp_den += (dp[file][mod][key]) ** 2
            l1_sp /= l1_sp_den
            l2_sp /= l2_sp_den

            l1_mp = 0
            l1_mp_den = 0
            l2_mp = 0
            l2_mp_den = 0
            for key in (mp_set | dp_set):
                l1_mp += abs(mp[file][mod][key] - dp[file][mod][key])
                l1_mp_den += abs(dp[file][mod][key])
                l2_mp += (mp[file][mod][key] - dp[file][mod][key]) ** 2
                l2_mp_den += (dp[file][mod][key]) ** 2
            l1_mp /= l1_mp_den
            l2_mp /= l2_mp_den

            adder(el1, l1, [l1_sp, l1_mp], e)
            adder(el2, l2, [l2_sp, l2_mp], e)
    precision = np.array(precision)
    recall = np.array(recall)
    accuracy = np.array(accuracy)
    l1 = np.array(l1)
    l2 = np.array(l2)
    write_set(precision, recall, accuracy, l1, l2, "statistics_set.csv")

    eprecision = np.array(eprecision)
    erecall = np.array(erecall)
    eaccuracy = np.array(eaccuracy)
    el1 = np.array(el1)
    el2 = np.array(el2)
    write_set(eprecision, erecall, eaccuracy, el1, el2, "estatistics_set.csv")


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
