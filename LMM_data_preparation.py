from enum import Enum
import os
import numpy as np
import pandas as pd
import json


class Algo(Enum):
    PCMCI = 2
    FPCMCI = 1
    doFPCMCI = 0
    
    
class Score(Enum):
    precision = 'precision'
    recall = 'recall'
    f1 = 'f1_score'
    shd = 'shd'
    fpr = 'fpr'
    nSpurious = 'nSpurious'
    
    
_PCMCI = 'pcmci'
_FPCMCI = 'fpcmci'
_doFPCMCI = 'dofpcmci'


def get_nvars_map(nvars):

    # Calculate the mean of the original list
    mean_value = np.mean(nvars)

    # Create the centered list
    return [x - mean_value for x in nvars]

    
def create_csv(nvars, resfolder):
    nvars = list(np.arange(nvars[0], nvars[1] + 1))
    res_path = os.getcwd() + "/results/" + resfolder + "/"
    nvars_map = get_nvars_map(nvars)
    
    for score in Score.__members__.values():
        
        # Dataframe initialisation
        # column_names = ['algo', 'nvars', score.value]
        column_names = ['algo', 'nconfounded', score.value]
        df = pd.DataFrame(columns = column_names)
        
        for idx, nv in enumerate(nvars):
            with open(res_path + str(nv) + ".json") as json_file:
                r = json.load(json_file)
                
                for i in r.keys():
                    if score == Score.nSpurious:
                        df.loc[len(df)] = [Algo.PCMCI.value, nv, r[i][_PCMCI]["N_SpuriousLinks"] / r[i]["N_ExpectedSpuriousLinks"]]
                        df.loc[len(df)] = [Algo.FPCMCI.value, nv, r[i][_FPCMCI]["N_SpuriousLinks"] / r[i]["N_ExpectedSpuriousLinks"]]
                        df.loc[len(df)] = [Algo.doFPCMCI.value, nv, r[i][_doFPCMCI]["N_SpuriousLinks"] / r[i]["N_ExpectedSpuriousLinks"]]
                    else:
                        df.loc[len(df)] = [Algo.PCMCI.value, nv, r[i][_PCMCI][score.value]]
                        df.loc[len(df)] = [Algo.FPCMCI.value, nv, r[i][_FPCMCI][score.value]]
                        df.loc[len(df)] = [Algo.doFPCMCI.value, nv, r[i][_doFPCMCI][score.value]]
                        
        # Data standardisation
        df.to_csv(res_path + score.value + '.csv', index=False)




if __name__ == '__main__':   

    # resfolder = ['nvariable_1hconf_nonlin_1000_1000_0_0.5']
    resfolder = ['nconfounded_nonlin_1000_1000_0_0.5']
    
    for r in resfolder:
        
        # create_csv([7, 14], r)
        create_csv([2, 7], r)
                