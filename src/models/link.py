"""
    src/models/link.py
    ------------------

    This module implements the `LinkStatePredictor`, a neural network–based 
    predictor for classifying the state of a wireless link between a UAV 
    and its receiver (e.g., LOS, NLOS). 

    The predictor combines learned representations of relative UAV–receiver 
    geometry (`dvec`) and categorical receiver types (`rx_type`) to 
    estimate the link state. Preprocessing steps include one-hot encoding 
    of receiver types, geometric feature transformation, and standard 
    normalization. 
"""
import pickle
import json
import numpy as np
import tensorflow as tf
tfk = tf.keras
tfkl, tfkm = tfk.layers, tfk.models

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.config.data import LinkState
from src.models.utilities.preproc import preproc_to_param, param_to_preproc, PreprocType

from pathlib import Path
from typing import Dict, List, Tuple, Union


class LinkStatePredictor:
    """
        A high-level predictor class for UAV–receiver link state 
        classification. Processing of geometric features and receiver.
        Construction of training of a neural network model.
    """
    def __init__(self,
        rx_types            : List[Union[str, int]],
        n_unit_links        : Tuple[int, ...],
        n_dimensions        : int   = 3,
        add_zero_los_frac   : float = 0.10,
        dropout_rate        : float = 0.15,
        seed                : int   = 42,
        directory           : str   = "link",
        **kwargs
    ):
        """ Initialize the LinkState Predictor """
        # Construct the Directory
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

        # Model and Preprocessors initialization
        self.model = None
        self.link_scaler = None
        self.rx_type_encoder = None

        # Parameters
        self.rx_types = list(rx_types)
        self.n_unit_links = tuple(n_unit_links)
        self.n_dimensions = int(n_dimensions)
        self.add_zero_los_frac = float(add_zero_los_frac)
        self.dropout_rate = float(dropout_rate)
        self.seed = int(seed)

        # Versions
        self.__version__ = 0
    

    def build(self):
        """
            Constructs the `tf.keras.Model` construction, this creates 
            a `Sequential` model architecture.

            Architecture:
            -------------
                - Input layer sized according to the `one-hot encoding`
                  receiver representation of the rx_types.
                - A configuration stack of fully connected dense,
                  (`hidden`) layers, each followed by a 
                  `BatchNormalization` for normalize potential outliers
                  avoiding misleading learning patterns. Followed 
                  thereafter by a activation `sigmoid`, with an optional
                  dropout layer.
                - A final softmax output layer with dimensionality equal
                  to the number of possible link-states 
                  (`LinkState.N_STATES`)
            
            Notes:
            ------
                - The input dimensionality is `2 · len(rx_types)`, as both
                  horizontal (dx) as well as the vertical (dz) distances
                  are broadcast across the receiver types.
                - This method must be called before `self.fit()`, 
                  `self.predict()` or `self.load_weights()`.
            -------
        """
        # Create the Input Layer
        self.model = tfkm.Sequential()
        self.model.add(tfkl.Input(shape=((2*len(self.rx_types)),), name='Input'))

        # Hidden Layers
        for i, units in enumerate(self.n_unit_links):
            name = f"Hidden-{i}"
            self.model.add(tfkl.Dense(units=units, name=name))
            self.model.add(tfkl.BatchNormalization())
            self.model.add(tfkl.Activation('sigmoid'))
            if self.dropout_rate > 0:
                self.model.add(tfkl.Dropout(self.dropout_rate))
        
        # Output Layer
        self.model.add(tfkl.Dense(LinkState.N_STATES, activation='softmax', name='Output'))
    

    def fit(self,
        dtr: Dict[str, np.ndarray], dts: Dict[str, np.ndarray],
        epochs          : int    = 50,
        batch_size      : int   = 512,
        learning_rate   : float = 1e-4
    ):
        """
            Train the `LinkStatePredictor` model on labeled data. This
            method train the model which unfortunately pass NumPy arrays
            for training (small model, is fine). 

            Args:
            -----
                dtr:    Training dataset dictionary containing `dvec`,
                        `rx_type`, and `link_state`.
                dts:    Validation dataset dictionary with the same keys
                        `dtr`, used for test_step for validating.
                epochs: int = 50, Number of full passes over the training
                        dataset.
                batch_size: int = 512, Number of samples per training
                            batch, `512` rather decent for utilitzation
                            of GPU, depending on hardware may need alter.
                learning_rate:  float = 1e-4, learning-rate for the applied
                                optimizer, this using Adam.
            
            Returns:
            --------
                history:    tf.keras.callbacks.History, training history
                            object with the losses/accuracy metrics.
            --------
            Notes:

                Automatically fits preprocessors (scaler, encoder) on\
                    training data, no need to bother.
                Augments training set with synthetic LOS-zero samples \
                    if enabled.
                Uses sparse categorical crossentropy loss and\
                    accuracy metric. \\
        """
        # Fit the encoder and prepare the data
        xtr, ytr = self._prepare_arrays(dtr, True)
        xts, yts = self._prepare_arrays(dts, False)

        # Compile the model
        self.model.compile(optimizer=tfk.optimizers.Adam(learning_rate=learning_rate),
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
        history = self.model.fit(xtr, ytr, batch_size=batch_size, epochs=epochs,
                                 validation_data=(xts, yts))
        self.history = history
        return history
    

    def predict(self, dvec: np.ndarray, rx_type: np.ndarray) -> np.ndarray:
        """
            Method for running interference on new input samples.
            Args:
            -----
            dvec:   np.ndarray, shape (N, 3) Relative UAV–
                    receiver displacement vectors.  
            rx_type:    np.ndarray, shape (N,) Receiver type 
                        labels (categorical).  

            Returns
            -------
            np.ndarray, shape (N, LinkState.N_STATES)
                Predicted class probabilities for each link state.  

            Notes
            -----
            - Input samples are transformed using the fitted encoder and scaler.  
            - Returns probabilities; `argmax` can be used to obtain hard predictions.  
        """
        return self.model.predict(self._transform_link(dvec, rx_type))
    


    def save(self):
        """
            Persist the trained model and preprocessing artifacts 
            to disk.

            Saves three files inside `self.directory`:
            - `preproc.pkl`:    pickled dictionary with fitted scaler, 
                                encoder, and config.  
            - `param.json`: JSON log of training history 
                            (loss/accuracy curves).  
            - `link.weights.h5`:    TensorFlow model weights in 
                                    HDF5 format.  
            Notes
            -----
            - Ensures reproducibility by recording framework version and parameters.  
            - Use `load()` to restore model and preprocessing state.  
        """
        payload = {
            'version': self.__version__,
            'framework': {'tensorflow': tf.__version__},
            'link_scaler': preproc_to_param(self.link_scaler, PreprocType.STANDARD),
            'rx_encoder': preproc_to_param(self.rx_type_encoder, PreprocType.ONE_HOT),
            'config': {
                'n_unit_links': self.n_unit_links,
                'rx_types': self.rx_types,
                'n_dimensions': self.n_dimensions,
                'add_zero_los_frac': self.add_zero_los_frac,
                'dropout_rate': self.dropout_rate,
                'seed': self.seed
            }
        }
        with open(self.directory / "preproc.pkl", 'wb') as fp:
            pickle.dump(payload, fp)
        with open(self.directory / "param.json", 'w') as fp:
            json.dump(self.history.history, fp)
        self.model.save_weights(str(self.directory / "link.weights.h5")) 


    def load(self):
        """
            Restore the model and preprocessing artifacts from 
            saved files.

            Loads:
            ------
            - `preproc.pkl` (scaler, encoder, config).  
            - `link.weights.h5` (trained model weights).  

            Raises
            ------
            FileNotFoundError:  If the required files are 
                                missing in the directory.  
            Warning:    If the saved version does not match 
                        `__version__`.  

            Notes
            -----
            - Rebuilds the Keras model before loading weights.  
            - Respects configuration values stored during training.  
        """
        preproc_path = self.directory / "preproc.pkl"
        weights_path = self.directory / "link.weights.h5"
        if not preproc_path.exists() or not weights_path.exists():
            raise FileNotFoundError("Model files not found for link")
        
        with open(preproc_path, 'rb') as fp:
            payload = pickle.load(fp)
        if payload.get("version", 0) != self.__version__:
            raise Warning("Version mismatch! might lead to incompatibility")
        
        self.link_scaler = param_to_preproc(param=payload['link_scaler'],
                                            proc_type=ProcType.STANDARD_SCALER)
        self.rx_type_encoder = param_to_preproc(param=payload['rx_encoder'],
                                                proc_type=ProcType.ONE_HOT_ENCODER)
        cfg = payload.get("config", {})
        self.n_unit_links = cfg.get("n_unit_link", self.n_unit_links)
        self.rx_types = cfg.get("rx_types", self.rx_types)
        self.add_zero_los_frac = cfg.get("add_zero_los_frac", self.add_zero_los_frac)
        self.dropout_rate = cfg.get("dropout_rate", self.dropout_rate)
        self.seed = cfg.get("seed", self.seed)

        self.build()
        self.model.load_weights(str(weights_path))

    # 

    def _prepare_arrays(self,
        data    : Dict[str, np.ndarray],
        fit     : bool  = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
            Transform raw dataset dictionary into model-ready arrays.

            Steps
            -----
            1.  Extract displacement vectors (`dvec`), receiver types 
                (`rx_type`), and link state labels (`link_state`).  
            2.  If `fit=True`, initialize and fit preprocessors on 
                training data.  
            3.  Apply LOS-zero augmentation (optional).  
            4.  Return transformed features `X` and labels `y`.  

            Args
            ----
            data :  Dict[str, np.ndarray]: Input dataset with keys
                    'dvec', 'rx_type', and 'link_state'.  
            fit :   bool, default=False, Whether to fit preprocessing 
                    steps (True for training, False for validation/test).  

            Returns
            -------
            x : np.ndarray. Feature matrix ready for model input.  
            y : np.ndarray, Corresponding target labels.  
        """
        dvec = np.asarray(data['dvec'])
        rx_type = np.asarray(data['rx_type'])
        y = np.asarray(data['link_state']) # Target Attribute

        if fit:
            # Fit the encoder and scaler on the raw data
            _ = self._transform_link(dvec, rx_type, True)

            # Augment training data
            dvec, rx_type, y = self._add_los_zero(dvec, rx_type, y)

            # Transform augmented training data
            x = self._transform_link(dvec, rx_type, False)
        else:
            dvec, rx_type, y = self._add_los_zero(dvec, rx_type, y)
            x = self._transform_link(dvec, rx_type, False)
        
        return x, y
    

    def _transform_link(self,
        dvec        : np.ndarray,
        rx_type     : np.ndarray,
        fit         : bool  = False
    ) -> np.ndarray:
        """
        Convert geometric and categorical features into standardized inputs.

        Steps
        -----
        1. Compute horizontal distance `dx = sqrt(x² + y²)` and vertical distance `dz`.  
        2. Encode receiver types into one-hot vectors.  
        3. Broadcast `dx` and `dz` across receiver encodings to produce per-type features.  
        4. Concatenate features horizontally and scale with `StandardScaler`.  

        Args
        ----
        dvec : np.ndarray, shape (N, 3)
            Relative UAV–receiver displacement vectors.  
        rx_type : np.ndarray, shape (N,)
            Receiver type identifiers.  
        fit : bool, default=False
            If True, fit preprocessors on input data.  

        Returns
        -------
        x : np.ndarray, shape (N, 2 * len(rx_types))
            Transformed and scaled feature matrix.  
        """
        # Compute distances
        dx = np.sqrt(dvec[:, 0]**2 + dvec[:, 1]**2)[:, None]
        dz = dvec[:, 2][:, None]

        # Encode RX-Types
        if fit:
            if self.rx_type_encoder is None:
                self.rx_type_encoder = OneHotEncoder(sparse_output=False,
                                                     handle_unknown='ignore',
                                                     min_frequency=None)
            rx_one = self.rx_type_encoder.fit_transform(rx_type[:, None])
        else:
            rx_one = self.rx_type_encoder.transform(rx_type[:, None])
        
        # Broadcast distances across one-hot encoder features
        x_dx = rx_one * dx
        x_dz = rx_one * dz

        # Concatenate horizontally (dx, dz) for each RX type
        x0 = np.hstack([x_dx, x_dz]).astype(np.float32)

        # Apply scaling
        if fit:
            if self.link_scaler is None:
                self.link_scaler = StandardScaler()
            return self.link_scaler.fit_transform(x0)
        return self.link_scaler.transform(x0)
    

    def _add_los_zero(self,
        dvec        : np.ndarray,
        rx_type     : np.ndarray,
        link_state  : np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
            Augment dataset with synthetic zero-distance LOS samples. 
            This augmentation helps improve robustness by injecting 
            near-ground-truth samples that represent UAV–receiver pairs 
            with minimal displacement and guaranteed LOS.

            Args
            ----
            dvec:   np.ndarray, shape (N, 3) Original displacement 
                    vectors.  
            rx_type:    np.ndarray, shape (N,), Receiver types.  
            link_state: np.ndarray, shape (N,) Ground-truth link 
                        state labels.

            Returns
            -------
            dvec_new : np.ndarray
                Augmented displacement vectors.  
            rx_type_new : np.ndarray
                Augmented receiver types.  
            link_state_new : np.ndarray
                Augmented link state labels.  

            Notes
            -----
            - Number of added samples is controlled by `self.add_zero_los_frac`.  
            - Augmented LOS samples have zero XY displacement and non-negative Z displacement.  
        """
        n_samples = dvec.shape[0]
        n_add = int(n_samples * self.add_zero_los_frac)
        if n_add <= 0:
            return dvec, rx_type, link_state
        
        # Random indices for augmentation
        indices = np.random.randint(n_samples, size=n_add)

        # Build new samples
        dvec_i = np.zeros((n_add, dvec.shape[1]), dtype=dvec.dtype)
        dvec_i[:, 2] = np.maximum(dvec[indices, 2], 0)

        rx_type_i = rx_type[indices]
        link_state_i = np.full(n_add, LinkState.LOS, dtype=link_state.dtype)

        # Concatenate
        dvec_new = np.concatenate((dvec, dvec_i), axis=0)
        rx_type_new = np.concatenate((rx_type, rx_type_i), axis=0)
        link_state_new = np.concatenate((link_state, link_state_i), axis=0)

        return dvec_new, rx_type_new, link_state_new