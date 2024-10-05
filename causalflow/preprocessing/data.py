"""
This module provides the Data class.

Classes:
    Data: public class for handling data used for the causal discovery.
"""

import numpy as np
import pandas as pd
from matplotlib import gridspec
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from causalflow.preprocessing.Subsampler import Subsampler
from causalflow.preprocessing.subsampling_methods.SubsamplingMethod import SubsamplingMethod


class Data():
    """Data class manages the preprocess of the data before the causal analysis."""
    
    def __init__(self, data, vars = None, fill_nan = True, stand = False, subsampling : SubsamplingMethod = None, show_subsampling = False):
        """
        Class constructor.

        Args:
            data (str/DataFrame/np.array): it can be a string specifing the path of a csv file to load/pandas.DataFrame/numpy.array.
            vars (list(str), optional): List containing variable names. If unset then, 
                if data = (str/DataFrame) vars = data columns name elif data = np.array vars = [X_0 .. X_N]
                Defaults to None.
            fill_nan (bool, optional): Fill NaNs bit. Defaults to True.
            stand (bool, optional): Standardization bit. Defaults to False.
            subsampling (SubsamplingMethod, optional): Subsampling method. If None not active. Defaults to None.
            show_subsampling (bool, optional): If True shows subsampling result. Defaults to False.

        Raises:
            TypeError: if data is not str - DataFrame - ndarray.
        """
        # Data handling
        if type(data) == np.ndarray:
            self.d = pd.DataFrame(data)
            if vars is None: self.d.columns = list(['X_' + str(f) for f in range(len(self.d.columns))])
        elif type(data) == pd.DataFrame:
            self.d = data
        elif type(data) == str:
            self.d = pd.read_csv(data)
        else:
            raise TypeError("data field not in the correct type\ndata must be one of the following type:\n- numpy.ndarray\n- pandas.DataFrame\n- .csv path")
            
        
        # Columns name handling
        if vars is not None:
            self.d.columns = list(vars)
                
        
        self.orig_features = self.features
        self.orig_pretty_features = self.pretty_features
        self.orig_N = self.N
        self.orig_T = len(self.d)

        # Filling NaNs
        if fill_nan:
            if self.d.isnull().values.any():
                self.d.fillna(inplace=True, method="ffill")
                self.d.fillna(inplace=True, method="bfill")

        # Subsampling data
        if subsampling is not None:
            subsampler = Subsampler(self.d, ss_method = subsampling)
            self.d = pd.DataFrame(subsampler.subsample(), columns = self.features)
            if show_subsampling: subsampler.plot_subsampled_data()

        # Standardize data
        if stand:
            scaler = StandardScaler()
            scaler = scaler.fit(self.d)
            self.d = pd.DataFrame(scaler.transform(self.d), columns = self.features)
        
    @property  
    def features(self):
        """
        Return list of features.

        Returns:
            list(str): list of feature names.
        """
        return list(self.d.columns)

    @property
    def pretty_features(self):
        """
        Return list of features with LATEX symbols.
                
        Returns:
            list(str): list of feature names.
        """
        return [r'$' + str(v) + '$' for v in self.d.columns]
    
    @property
    def N(self):
        """
        Number of features.
        
        Returns:
            (int): number of features.
        """
        return len(self.d.columns)

    @property
    def T(self):
        """
        Dataframe length.
        
        Returns:
            (int): dataframe length.
        """
        return len(self.d)
                       
            
    def shrink(self, selected_features):
        """
        Shrink dataframe d on the selected features.

        Args:
            selected_features (list(str)): list of variables.
        """
        self.d = self.d[selected_features]
        
                    
    def plot_timeseries(self, savefig = None):
        """
        Plot timeseries data.
        
        Args:
            savefig (str): figure path.
        """
        # Create grid
        gs = gridspec.GridSpec(self.N, 1)

        # Time vector
        T = list(range(self.T))

        plt.figure()
        for i in range(0, self.d.shape[1]):
            ax = plt.subplot(gs[i, 0])
            plt.plot(T, self.d.values[:, i], color = 'tab:red')
            plt.ylabel(str(self.pretty_features[i]))

        if savefig is not None:
            plt.savefig(savefig)
        else:
            plt.show()
            
            
    def save_csv(self, csvpath):
        """
        Save timeseries data into a CSV file.
        
        Args:
            csvpath (str): CSV path.
        """
        self.d.to_csv(csvpath, index=False)