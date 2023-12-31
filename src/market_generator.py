import datetime
import numpy as np
import torch
import yfinance as yf
from esig import tosig
import signatory
import iisignature
from sklearn.preprocessing import MinMaxScaler

from utils.leadlag import leadlag
from cvae import CVAE
from rough_bergomi import rough_bergomi

class MarketGenerator:
    def __init__(self, ticker, start=datetime.date(2000, 1, 1),
                 end=datetime.date(2019, 1, 1), freq="M",
                 sig_order=4, rough_bergomi=None):

        self.ticker = ticker
        self.start = start
        self.end = end
        self.freq = freq
        self.order = sig_order

        if rough_bergomi:
             self._load_rough_bergomi(rough_bergomi)
        else:
            self._load_data()

        self._build_dataset()

        # Set it to None and assign it later
        self.generator = None

    def _load_rough_bergomi(self, params):
        grid_points_dict = {"M": 28, "W": 5, "Y": 252}
        grid_points = grid_points_dict[self.freq]
        params["T"] = grid_points / grid_points_dict["Y"]

        paths = rough_bergomi(grid_points, **params)

        self.windows = [leadlag(path) for path in paths]


    def _load_data(self):
        try:
            # pdr.get_data_yahoo is not available
            # self.data = pdr.get_data_yahoo(self.ticker, self.start, self.end)["Close"]
            self.data = yf.download(self.ticker, self.start, self.end)
            print("Download data successfully")
            print("Shape of downloaded data:{}".format(self.data.shape))            
        except:
            raise RuntimeError(f"Could not download data for {self.ticker} from {self.start} to {self.end}.")

        self.windows = []

        # Apply lead-lag transformation
        print("Apply lead lag transformation...")
        for idx,(_, window) in enumerate(self.data.resample(self.freq)):

            # Extract data
            # values shape be like: (20,6), (19,6), (23,6),...
            values = window.values

            # Check NaN
            if np.isnan(np.min(values)):
                print("Has NaN!")
                print(values)
                break

            # Lead lag transform
            # transformed path.shape be like: (39, 12), (41,12), etc
            if idx == 0:
                print("Shape before lead lag:{}".format(values.shape))
            path = leadlag(values)
            if idx == 0:
                print("Shape after lead lag:{}".format(path.shape))
            path_shape = path.shape

            # To fit signatory input format
            # input shape: (batch,stream,channel)
            # set batch size == 1
            # path = torch.from_numpy(path).float().view(1,path_shape[0],path_shape[1])
            self.windows.append(path)

        print("windows length:{}".format(len(self.windows)))
        print("element shape:{}".format(self.windows[0].shape))

    # def _logsig(self, path):
    #     return tosig.stream2logsig(path, self.order)

    def _build_dataset(self):
        if self.order:
            print("Calculate signatures...")
            print("We dropout the signatures with nan.")
            self.orig_logsigs = []
            self.normalized_logsigs = []
            self.scaler = MinMaxScaler(feature_range=(0.00001, 0.99999))
            for path in self.windows:

                # Compute signatures
                # sig = signatory.signature(path,self.order)
                s = iisignature.prepare(12,self.order)
                logsig = iisignature.logsig(path,s)
                self.orig_logsigs.append(logsig)

            self.orig_logsigs = np.array(self.orig_logsigs)

        normalized_logsigs = self.orig_logsigs
        normalized_logsigs = self.scaler.fit_transform(normalized_logsigs)

        self.logsig = np.array(normalized_logsigs)[1:]
        self.conditions = np.array(normalized_logsigs)[:-1]

        print("self.logsig shape:{}".format(self.logsig.shape))
        print("self.conditions shape:{}".format(self.conditions.shape))

    def train(self, n_epochs=10000):
        self.generator = CVAE(data=self.logsig, data_cond=self.conditions, latent_dim=8, alpha=0.003)
        self.generator.train(n_epochs=n_epochs)

    def generate(self, logsig, n_samples=1, normalised=False):
        generated = self.generator.generate(logsig, n_samples=n_samples)

        if normalised:
            return generated

        if n_samples is None:
            return self.scaler.inverse_transform(generated.reshape(1, -1))[0]

        return self.scaler.inverse_transform(generated)
