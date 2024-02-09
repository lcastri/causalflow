from enum import Enum
import os
import numpy as np
import pandas as pd
import json
from res_statistics_new import *


def create_csv(nvars, resfolder):
    since = datetime.datetime(1900, 1, 1, 0, 0, 0, 0)
    nvars = list(np.arange(nvars[0], nvars[1] + 1))
    res_path = os.getcwd() + "/results/" + resfolder + "/"
    # nvars_map = get_nvars_map(nvars)
    
    for score in Metric.__members__.values():
        
        # Dataframe initialisation
        column_names = ['algo', 'nvars', score.value]
        # column_names = ['algo', 'nconfounded', score.value]
        df = pd.DataFrame(columns = column_names)
        
        for idx, nv in enumerate(nvars):
            with open(res_path + str(nv) + ".json") as json_file:
                r = json.load(json_file)
                
                for i in r.keys():
                    if score == Metric.TIME:
                        t = datetime.datetime.strptime(r[i][Algo.PCMCI.value][Metric.TIME.value], '%H:%M:%S.%f')
                        df.loc[len(df)] = [Algo.PCMCI.value, nv, (t - since).total_seconds()]
                        t = datetime.datetime.strptime(r[i][Algo.FPCMCI.value][Metric.TIME.value], '%H:%M:%S.%f')
                        df.loc[len(df)] = [Algo.FPCMCI.value, nv, (t - since).total_seconds()]
                        t = datetime.datetime.strptime(r[i][Algo.CAnDOIT.value][Metric.TIME.value], '%H:%M:%S.%f')
                        df.loc[len(df)] = [Algo.CAnDOIT.value, nv, (t - since).total_seconds()]
                    elif score == Metric.N_ESPU:
                        if r[i][jWord.N_GSPU.value] != 0:
                            df.loc[len(df)] = [Algo.PCMCI.value, nv, r[i][Algo.PCMCI.value][Metric.N_ESPU.value] / r[i][jWord.N_GSPU.value]]
                            df.loc[len(df)] = [Algo.FPCMCI.value, nv, r[i][Algo.FPCMCI.value][Metric.N_ESPU.value] / r[i][jWord.N_GSPU.value]]
                            df.loc[len(df)] = [Algo.CAnDOIT.value, nv, r[i][Algo.CAnDOIT.value][Metric.N_ESPU.value] / r[i][jWord.N_GSPU.value]]
                        else:
                            df.loc[len(df)] = [Algo.PCMCI.value, nv, 0]
                            df.loc[len(df)] = [Algo.FPCMCI.value, nv, 0]
                            df.loc[len(df)] = [Algo.CAnDOIT.value, nv, 0]
                    else:
                        df.loc[len(df)] = [Algo.PCMCI.value, nv, r[i][Algo.PCMCI.value][score.value]]
                        df.loc[len(df)] = [Algo.FPCMCI.value, nv, r[i][Algo.FPCMCI.value][score.value]]
                        df.loc[len(df)] = [Algo.CAnDOIT.value, nv, r[i][Algo.CAnDOIT.value][score.value]]
                        
        # Data standardisation
        df.to_csv(res_path + score.value + '.csv', index=False)




if __name__ == '__main__':   

    resfolder = ['rebuttal/nvariable_nonlin_1250_250']
    # resfolder = ['rebuttal/nconfounded_nonlin_1250_250']
    
    for r in resfolder:
        
        create_csv([7, 14], r)
        # create_csv([0, 7], r)
                