import datetime
from enum import Enum
import os
import numpy as np
import pandas as pd
import json
from AIS_plotting import Metric, Algo
from sklearn.utils import resample


def bootstrap_data(df, n_bootstrap_samples):
    bootstrapped_data = []
    for _ in range(n_bootstrap_samples):
        bootstrapped_sample = resample(df, replace=True)
        bootstrapped_data.append(bootstrapped_sample)
    return pd.concat(bootstrapped_data)


def create_csv(nvars, resfolder, extrafield = False):
    since = datetime.datetime(1900, 1, 1, 0, 0, 0, 0)
    nvars = list(np.arange(nvars[0], nvars[1] + 1))
    res_path = os.getcwd() + "/results/" + resfolder + "/"
    
    for score in Metric.__members__.values():
        
        column_names = ['algo', 'nvars', score.value] if not extrafield else ['algo', 'nint', score.value]
        df = pd.DataFrame(columns = column_names)
        
        for _, nv in enumerate(nvars):
            json_name = res_path + str(nv) + ".json" if not extrafield else res_path + resfolder[-2:] + ".json"
            with open(json_name) as json_file:
                r = json.load(json_file)
                
                for i in r.keys():

                    if not extrafield:
                        candoit_keys = [key for key in r[i] if key.startswith('candoit__')]
                        selected_candoit = candoit_keys[np.argmax([r[i][k]['graph_f1_score'] for k in candoit_keys])]
                        if score == Metric.TIME:
                            # CAnDOIT best
                            t = datetime.datetime.strptime(r[i][selected_candoit][score.value], '%H:%M:%S.%f')
                            df.loc[len(df)] = ["candoit_best", nv, (t - since).total_seconds()]
                            # CAnDOIT mean
                            ts = [datetime.datetime.strptime(r[i][candoit_run][score.value], '%H:%M:%S.%f') for candoit_run in candoit_keys]
                            tmp = np.mean([(t - since).total_seconds() for t in ts])
                            df.loc[len(df)] = ["candoit_mean", nv, tmp]
                            # LPCMCI 
                            t = datetime.datetime.strptime(r[i][Algo.LPCMCI.value][Metric.TIME.value], '%H:%M:%S.%f')
                            df.loc[len(df)] = [Algo.LPCMCI.value, nv, (t - since).total_seconds()]
                        elif score == Metric.PAGSIZE:
                            # CAnDOIT best
                            df.loc[len(df)] = ["candoit_best", nv, np.log10(r[i][selected_candoit][score.value])]
                            # CAnDOIT mean
                            tmp = np.mean([np.log10(r[i][candoit_run][score.value]) for candoit_run in candoit_keys])
                            df.loc[len(df)] = ["candoit_mean", nv, tmp]
                            # LPCMCI 
                            df.loc[len(df)] = [Algo.LPCMCI.value, nv, np.log10(r[i][Algo.LPCMCI.value][score.value])]
                        else:
                            # CAnDOIT best
                            df.loc[len(df)] = ["candoit_best", nv, r[i][selected_candoit][score.value]]
                            # CAnDOIT mean
                            tmp = np.mean([r[i][candoit_run][score.value] for candoit_run in candoit_keys])
                            df.loc[len(df)] = ["candoit_mean", nv, tmp]
                            # LPCMCI 
                            df.loc[len(df)] = [Algo.LPCMCI.value, nv, r[i][Algo.LPCMCI.value][score.value]]
                    else:
                        candoit_keys = [key for key in r[i][str(nv)] if key.startswith('candoit__')]
                        selected_candoit = candoit_keys[np.argmax([r[i][str(nv)][k]['graph_f1_score'] for k in candoit_keys])]
                        if score == Metric.TIME:
                            # CAnDOIT best
                            t = datetime.datetime.strptime(r[i][str(nv)][selected_candoit][score.value], '%H:%M:%S.%f')
                            df.loc[len(df)] = ["candoit_best", nv, (t - since).total_seconds()]
                            # CAnDOIT mean
                            ts = [datetime.datetime.strptime(r[i][str(nv)][candoit_run][score.value], '%H:%M:%S.%f') for candoit_run in candoit_keys]
                            tmp = np.mean([(t - since).total_seconds() for t in ts])
                            df.loc[len(df)] = ["candoit_mean", nv, tmp]
                            # LPCMCI 
                            t = datetime.datetime.strptime(r[i][Algo.LPCMCI.value][Metric.TIME.value], '%H:%M:%S.%f')
                            df.loc[len(df)] = [Algo.LPCMCI.value, nv, (t - since).total_seconds()]
                        elif score == Metric.PAGSIZE:
                            # CAnDOIT best
                            df.loc[len(df)] = ["candoit_best", nv, np.log10(r[i][str(nv)][selected_candoit][score.value])]
                            # CAnDOIT mean
                            tmp = np.mean([np.log10(r[i][str(nv)][candoit_run][score.value]) for candoit_run in candoit_keys])
                            df.loc[len(df)] = ["candoit_mean", nv, tmp]
                            # LPCMCI 
                            df.loc[len(df)] = [Algo.LPCMCI.value, nv, np.log10(r[i][Algo.LPCMCI.value][score.value])]
                        else:
                            # CAnDOIT best
                            df.loc[len(df)] = ["candoit_best", nv, r[i][str(nv)][selected_candoit][score.value]]
                            # CAnDOIT mean
                            tmp = np.mean([r[i][str(nv)][candoit_run][score.value] for candoit_run in candoit_keys])
                            df.loc[len(df)] = ["candoit_mean", nv, tmp]
                            # LPCMCI 
                            df.loc[len(df)] = [Algo.LPCMCI.value, nv, r[i][Algo.LPCMCI.value][score.value]]
                        
        bootstrapped_data = bootstrap_data(df, n_bootstrap_samples)
        bootstrapped_data.to_csv(res_path + score.value + '_boot.csv', index=False)




if __name__ == '__main__':   
    n_bootstrap_samples = 1000
    resfolder = 'AIS_major/AIS_major_S4'    
    create_csv([5, 12], resfolder)
    # create_csv([1, 3], resfolder, extrafield=True)
                