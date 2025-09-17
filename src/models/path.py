"""
    src/models/path.py
    -------------------
    This script contains a model `PathModel` which is the
    second stage in the `ChannelModel`. The PathModel receives
    the state of link (`NO_LINK`, `LOS`, `NLOS`)  from the 
    `LinkStatePredictor` and transform the data, final stage
    before passing the data to the Generative AI model it 
    converts the data structure from NumPy arrays to 
    TF Tensors.
"""
import pickle

import numpy as np
import tensorflow as tf
tfk = tf.keras

from pathlib import Path
from typing import Dict, List, Union

from src.config.model import get_config
from src.config.data import AngleIndex, LinkState
from src.config.const import (
    LIGHT_SPEED,
    PREPROCESSOR_FN
)

from src.math.cart_sph import cartesian_to_spherical, add_angles, sub_angles
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler


class PathModel:
    """
        The path-model operates at far-end of the Channel Model object
        and manages preprocessing the path-data (`path-loss`, `angles`,
        `delays`) and manages the `Gen-AI: Model` to which it pass the 
        transformed data to.
    """
    ANGLE_SCALE = 180.0  # degrees normalization constant

    def __init__(self,
        directory       : Union[str, Path],
        model_type      : str,
        rx_types        : List[Union[str, int]],
        n_max_paths     : int,
        max_pathloss    : float
    ):
        """
            Initialize the Path-Model Instance
        """
        # Directory Creations
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

        self.model_type = model_type.lower()
        self.cfg = get_config(self.model_type)
        self.model = None

        self.rx_types = list(rx_types)
        self.n_max_paths = int(n_max_paths)
        self.max_pathloss = float(max_pathloss)

        self._initialize_preprocessors()
    

    # ---------- Model Constructions ---------- #

    def build(self):
        """
            Constructs the generative model, configurations is retrieved 
            from `get_config()` implemented at `src/config/model` along
            with the configurations. Supported models are
                - VAE
            The gen model is being stored in `self.model`
        """
        if self.model_type == "vae":
            from src.models.generators.vae import Vae
            self.model = Vae(
                n_latent        = self.cfg.n_latent,
                n_data          = self.n_max_paths * (2 + AngleIndex.N_ANGLES),
                n_conditions    = 3 + len(self.rx_types),
                encoder_layers  = self.cfg.encoder_layers,
                decoder_layers  = self.cfg.decoder_layers,
                min_variance    = self.cfg.min_variance,
                dropout_rate    = self.cfg.dropout_rate,
                beta            = self.cfg.beta,
                beta_annealing_step = self.cfg.beta_annealing_step,
                kl_warmup_steps = self.cfg.kl_warmup_steps,
                init_kernel_std = self.cfg.init_kernel_std,
                init_bias_std   = self.cfg.init_bias_std,
                n_sort  = self.n_max_paths
            )
        else:
            raise ValueError(f"Unsupported model-type: '{self.model_type}'")
        self.model.run_eagerly = True
    

    def fit(self,
        dtr: Dict[str, np.ndarray], dts: Dict[str, np.ndarray],
        epochs: int = 100, batch_size: int = 512, 
        learning_rate: float = 1e-4
    ) -> tfk.callbacks.History:
        """
            Fit the generative model.

            Parameters
            ----------
            dtr : dict
                Training dataset
            dts : dict
                Validation dataset
            epochs : int
            batch_size : int
            learning_rate : float

            Returns
            -------
            tf.keras.callbacks.History
        """
        xtr = self._prepare_dataset(dtr, batch_size=batch_size, fit=True)
        xts = self._prepare_dataset(dts, batch_size=batch_size, fit=False)

        self.model.compile(optimizer=tfk.optimizers.Adam(learning_rate=learning_rate,
                                                         clipvalue=1.0))
        history = self.model.fit(x=xtr, validation_data=xts, epochs=epochs)
        self.history = history
        return history



# ------------------------------------ I/O ------------------------------------ #

    def save(self):
        """
        """
        if self.model is None:
            raise ValueError("Model is not yet built, call `self.build()` first")
        
        self.model.save(self.directory)
        
        from src.models.utils.preproc import preproc_to_param, ProcType
        with open(self.directory / PREPROCESSOR_FN, "wb") as fp:
            pickle.dump({
                "pathloss_scaler": preproc_to_param(self.pathloss_scaler,
                                                    ProcType.MIN_MAX_SCALER),
                "condition_scaler": preproc_to_param(self.condition_scaler,
                                                     ProcType.STANDARD_SCALER),
                "rx_encoder":   preproc_to_param(self.rx_encoder, 
                                                 ProcType.ONE_HOT_ENCODER),
                "delay_scale": float(self.delay_scale),
                "n_max_paths": int(self.n_max_paths),
                "max_pathloss": float(self.max_pathloss),
                "rx_types": list(self.rx_types)
            }, fp, pickle.HIGHEST_PROTOCOL)
    

    def load(self):
        """
        """
        from src.models.utils.preproc import param_to_preproc, ProcType
        path = self.directory / PREPROCESSOR_FN
        if not path.exists():
            raise FileNotFoundError("The loaded model does not exist")
        
        with open(path, "rb") as fp:
            params = pickle.load(fp)
        
        self.pathloss_scaler = param_to_preproc(params["pathloss_scaler"],
                                                ProcType.MIN_MAX_SCALER)
        self.condition_scaler = param_to_preproc(params["condition_scaler"],
                                                 ProcType.STANDARD_SCALER)
        self.rx_encoder = param_to_preproc(params["rx_encoder"],
                                           ProcType.ONE_HOT_ENCODER)
        self.delay_scale = float(params.get("delay_scale", 1.0))

        # Restoring the storage values 
        self.n_max_paths = int(params.get("n_max_paths", self.n_max_paths))
        self.max_pathloss = float(params.get("max_pathloss", self.max_pathloss))
        self.rx_types = list(params.get("rx_types", self.rx_types))

        self.build()
        self.model.load(self.directory)


    # --------------------------- Dataset Preparations --------------------------- #

    def _initialize_preprocessors(self):
        """
        """
        self.pathloss_scaler    = MinMaxScaler()
        self.condition_scaler   = StandardScaler()
        self.rx_encoder    = OneHotEncoder(sparse_output  = False,
                                           handle_unknown = 'ignore',
                                           min_frequency  = None)
        self.delay_scale = 1.0


    # def _prepare_dataset(self,
    #     data        : Dict[str, np.ndarray],
    #     batch_size  : int,
    #     fit         : bool = False
    # ) -> tf.data.Dataset:
    #     """
    #     """
    #     link_state = data["link_state"]
    #     valid_mask = (link_state != LinkState.NO_LINK)
    #     if not np.any(valid_mask): 
    #         raise ValueError("No valid links found in input data")
        
    #     # Filter data
    #     idx = np.flatnonzero(valid_mask)
    #     dvec = np.asarray(data["dvec"][idx], dtype=np.float32)
    #     rx = np.asarray(data["rx_type"][idx])
    #     los = np.asarray(link_state[idx] == LinkState.LOS, dtype=np.float32)

    #     try:
    #         # Conditions
    #         u = self._transform_conditions(dvec, rx, los, fit)

    #         # Path - Data
    #         pathloss = self._transform_pathloss(data["nlos_pl"][idx], fit)
    #         angles = self._transform_angles(dvec, data["nlos_ang"][idx], 
    #                                         data["nlos_pl"][idx])
    #         delays = self._transform_delays(dvec, data["nlos_dly"][idx], fit)
    #         x = np.concatenate([pathloss, angles, delays], axis=1).astype(np.float32)
    #     except Exception as e:
    #         raise RuntimeError(f"PathModel._prepare_dataset failed. "
    #                            f"Shapes: dvec={dvec.shape}, rx={rx.shape}, "
    #                            f"los={los.shape}. Error: {e}") from e

    #     if u.shape[0] != x.shape[0]:
    #         raise ValueError(f"Conditional / path sample mismatch: "
    #                          f"{u.shape[0]} vs {x.shape[0]}")

    #     dataset = tf.data.Dataset.from_tensor_slices((x, u))
    #     return dataset.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)

    def _prepare_dataset(
        self,
        data: Dict[str, np.ndarray],
        batch_size: int,
        fit: bool = False
    ) -> tf.data.Dataset:
        """
        Prepare TensorFlow dataset from raw input dictionary.

        Parameters
        ----------
        data : dict
            Dictionary containing "dvec", "rx_type", "link_state", 
            "nlos_pl", "nlos_ang", "nlos_dly".
        batch_size : int
        fit : bool
            Whether to fit scalers/encoders.

        Returns
        -------
        tf.data.Dataset
        """
        link_state = data["link_state"]
        valid_mask = (link_state != LinkState.NO_LINK)
        if not np.any(valid_mask): 
            raise ValueError("No valid links found in input data")

        # Filter valid samples
        idx = np.flatnonzero(valid_mask)
        dvec = np.asarray(data["dvec"][idx], dtype=np.float32)
        rx = np.asarray(data["rx_type"][idx])
        los = np.asarray(link_state[idx] == LinkState.LOS, dtype=np.float32)

        # Conditions + Features
        u = self._transform_conditions(dvec, rx, los, fit=fit)
        x = self._transform_data(dvec,
                                 nlos_pathloss=data["nlos_pl"][idx],
                                 nlos_angles=data["nlos_ang"][idx],
                                 nlos_delays=data["nlos_dly"][idx],
                                 fit=fit)

        if u.shape[0] != x.shape[0]:
            raise ValueError(f"Conditional / path sample mismatch: "
                             f"{u.shape[0]} vs {x.shape[0]}")

        dataset = tf.data.Dataset.from_tensor_slices((x, u))
        return dataset.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)

    
    # ----------------------- Transforming data/conditions ----------------------- #

    def _transform_conditions(self,
        dvec: np.ndarray, rx_types: np.ndarray, los: np.ndarray,
        fit: bool   = False
    ) -> np.ndarray:
        """
        """
        # Compute distances
        d3d = np.maximum(np.linalg.norm(dvec, axis=1), 1.0)
        dh = dvec[:, 2]

        base = np.empty((dvec.shape[0], 4), dtype=np.float64)
        base[:, 0] = d3d
        base[:, 1] = np.log10(d3d)
        base[:, 2] = dh
        base[:, 3] = los.astype(np.float64)

        if fit: rx_one = self.rx_encoder.fit_transform(rx_types.reshape(-1, 1))
        else: rx_one = self.rx_encoder.transform(rx_types.reshape(-1, 1))

        if rx_one.shape[1] > 1: rx_one = rx_one[:, :-1] # Drop last one

        # Concatenate
        cond = np.concatenate((base, rx_one), axis=1)
        if fit: cond = self.condition_scaler.fit_transform(cond)
        else: cond = self.condition_scaler.transform(cond)

        return cond.astype(np.float32, copy=False)


    # def _transform_data(self, 
    #   dvec: np.ndarray, 
    #   nlos_pathloss: np.ndarray, 
    #   nlos_angles: np.ndarray, 
    #   nlos_delays: np.ndarray, 
    #   fit: bool = False
    # ) -> np.ndarray:
    #     """
    #       Transform path data (pathloss, angles, delays) into model input tensor.
    #       """
    #     pathloss = self._transform_pathloss(nlos_pathloss, fit)
    #     angles = self._transform_angles(dvec, nlos_angles, nlos_pathloss)
    #     delays = self._transform_delays(dvec, nlos_delays, fit)

    #     # Concatenate along feature axis
    #     return np.hstack((pathloss, angles, delays)).astype(np.float32, copy=False)


    # def _inverse_transform_data(self, dvec: np.ndarray, x: np.ndarray):
    #     nmp = self.n_max_paths
    #     n_angles = 4 * nmp
    #     pathloss = x[:, :nmp]
    #     angles = x[:, nmp:nmp + n_angles]
    #     delays = x[:, nmp + n_angles:]
    #     return (
    #         self._inverse_transform_pathloss(pathloss),
    #         self._inverse_transform_angles(dvec, angles),
    #         self._inverse_transform_delays(dvec, delays)
    #     )


    def _transform_data(
        self,
        dvec: np.ndarray,
        nlos_pathloss: np.ndarray,
        nlos_angles: np.ndarray,
        nlos_delays: np.ndarray,
        fit: bool = False
    ) -> np.ndarray:
        """Transform path data (pathloss, angles, delays) into normalized model inputs."""
        pathloss = self._transform_pathloss(nlos_pathloss, fit=fit)
        angles   = self._transform_angles(dvec, nlos_angles, nlos_pathloss)
        delays   = self._transform_delays(dvec, nlos_delays, fit=fit)
        return np.hstack((pathloss, angles, delays)).astype(np.float32, copy=False)

    def _inverse_transform_data(
        self,
        dvec: np.ndarray,
        x: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Inverse transform model outputs into physical domain (pathloss, angles, delays)."""
        nmp = self.n_max_paths
        n_angles = 4 * nmp
        pathloss = x[:, :nmp]
        angles   = x[:, nmp:nmp + n_angles]
        delays   = x[:, nmp + n_angles:]
        return (
            self._inverse_transform_pathloss(pathloss),
            self._inverse_transform_angles(dvec, angles),
            self._inverse_transform_delays(dvec, delays)
        )

    # ------------------------- Path loss transformation ------------------------- #

    def _transform_pathloss(self,
        nlos_pathloss: np.ndarray, fit: bool = False
    ) -> np.ndarray:
        """
        """
        x0 = self.max_pathloss - nlos_pathloss[:, :self.n_max_paths]
        if fit: return self.pathloss_scaler.fit_transform(x0)
        return self.pathloss_scaler.transform(x0)
    
    
    def _inverse_transform_pathloss(self,
        pathloss: np.ndarray, 
        fit: bool = False
    ) -> np.ndarray:
        """
        """
        x0 = np.clip(pathloss, 0.0, 1.0)
        x0 = self.pathloss_scaler.inverse_transform(x0)
        x0 = np.fliplr(np.sort(x0, axis=-1))
        return self.max_pathloss - x0
    

    # --------------------------- Angles transformation --------------------------- #

    # def _transform_angles(self,
    #     dvec            : np.ndarray, 
    #     nlos_angles     : np.ndarray, 
    #     nlos_pathloss   : np.ndarray
    # ) -> np.ndarray:
    #     """
    #     """
    #     # Compute LOS angles
    #     _, los_aoa_phi, los_aoa_theta = cartesian_to_spherical(-dvec)
    #     _, los_aod_phi, los_aod_theta = cartesian_to_spherical(dvec)

    #     # Compute relative relative angles
    #     aoa_phi_rel, aoa_theta_rel = sub_angles(nlos_angles[..., AngleIndex.AOA_PHI],
    #                                             nlos_angles[..., AngleIndex.AOA_THETA],
    #                                             los_aoa_phi[:, None],
    #                                             los_aoa_theta[:, None])

    #     aod_phi_rel, aod_theta_rel = sub_angles(nlos_angles[..., AngleIndex.AOD_PHI],
    #                                             nlos_angles[..., AngleIndex.AOD_THETA],
    #                                             los_aod_phi[:, None],
    #                                             los_aod_theta[:, None])
        
    #     # Stack as [aoa_phi | aoa_theta | aod_phi | aod_theta], normalized 180
    #     nmp = self.n_max_paths
    #     out = np.empty((dvec.shape[0], 4 * nmp), dtype=np.float32)

    #     out[:, 0:nmp]       = aoa_phi_rel   / 180.0
    #     out[:, nmp:2*nmp]   = aoa_theta_rel / 180.0
    #     out[:, 2*nmp:3*nmp] = aod_phi_rel   / 180.0
    #     out[:, 3*nmp:4*nmp] = aod_theta_rel / 180.0

    #     return out
    

    # def _inverse_transform_angles(self,
    #     dvec: np.ndarray, angles: np.ndarray
    # ) -> np.ndarray:
    #     """
    #     """
    #     nmp = self.n_max_paths
    #     aoa_phi_rel     = angles[:, 0:nmp]          * 180.0
    #     aoa_theta_rel   = angles[:, nmp:2*nmp]      * 180.0
    #     aod_phi_rel     = angles[:, 2*nmp:3*nmp]    * 180.0
    #     aod_theta_rel   = angles[:, 3*nmp:4*nmp]    * 180.0

    #     # Compute LOS angles (cartesian to polar coordinates)
    #     _, los_aoa_phi, los_aoa_theta = cartesian_to_spherical(dvec)
    #     _, los_aod_phi, los_aod_theta = cartesian_to_spherical(-dvec)

    #     # Compute relative angles into absolute angles
    #     nlos_aoa_phi, nlos_aoa_theta = add_angles(aoa_phi_rel, aoa_theta_rel,
    #                                               los_aoa_phi[:, None],
    #                                               los_aoa_theta[:, None])

    #     nlos_aod_phi, nlos_aod_theta = add_angles(aod_phi_rel, aod_theta_rel,
    #                                               los_aod_phi[:, None],
    #                                               los_aod_theta[:, None])

    #     out = np.zeros((dvec.shape[0], nmp, AngleIndex.N_ANGLES))
    #     out[..., AngleIndex.AOA_PHI]    = nlos_aoa_phi
    #     out[..., AngleIndex.AOA_THETA]  = nlos_aoa_theta
    #     out[..., AngleIndex.AOD_PHI]    = nlos_aod_phi
    #     out[..., AngleIndex.AOD_THETA]  = nlos_aod_theta
    #     return out


    def _transform_angles(self, 
        dvec: np.ndarray, 
        nlos_angles: np.ndarray, 
        nlos_pathloss: np.ndarray
    ) -> np.ndarray:
        """
        """
        _, los_aoa_phi, los_aoa_theta = cartesian_to_spherical(-dvec)
        _, los_aod_phi, los_aod_theta = cartesian_to_spherical(dvec)

        aoa_phi_rel, aoa_theta_rel = sub_angles(nlos_angles[..., AngleIndex.AOA_PHI],
                                                nlos_angles[..., AngleIndex.AOA_THETA],
                                                los_aoa_phi[:, None],
                                                los_aoa_theta[:, None])

        aod_phi_rel, aod_theta_rel = sub_angles(nlos_angles[..., AngleIndex.AOD_PHI],
                                                nlos_angles[..., AngleIndex.AOD_THETA],
                                                los_aod_phi[:, None],
                                                los_aod_theta[:, None])

        nmp = self.n_max_paths
        out = np.empty((dvec.shape[0], 4 * nmp), dtype=np.float32)
        out[:, 0:nmp]       = aoa_phi_rel   / self.ANGLE_SCALE
        out[:, nmp:2*nmp]   = aoa_theta_rel / self.ANGLE_SCALE
        out[:, 2*nmp:3*nmp] = aod_phi_rel   / self.ANGLE_SCALE
        out[:, 3*nmp:4*nmp] = aod_theta_rel / self.ANGLE_SCALE
        return out

    def _inverse_transform_angles(self, dvec: np.ndarray, angles: np.ndarray) -> np.ndarray:
        nmp = self.n_max_paths
        aoa_phi_rel     = angles[:, 0:nmp]          * self.ANGLE_SCALE
        aoa_theta_rel   = angles[:, nmp:2*nmp]      * self.ANGLE_SCALE
        aod_phi_rel     = angles[:, 2*nmp:3*nmp]    * self.ANGLE_SCALE
        aod_theta_rel   = angles[:, 3*nmp:4*nmp]    * self.ANGLE_SCALE

        _, los_aoa_phi, los_aoa_theta = cartesian_to_spherical(dvec)
        _, los_aod_phi, los_aod_theta = cartesian_to_spherical(-dvec)

        nlos_aoa_phi, nlos_aoa_theta = add_angles(aoa_phi_rel, aoa_theta_rel,
                                                  los_aoa_phi[:, None],
                                                  los_aoa_theta[:, None])
        nlos_aod_phi, nlos_aod_theta = add_angles(aod_phi_rel, aod_theta_rel,
                                                  los_aod_phi[:, None],
                                                  los_aod_theta[:, None])

        out = np.zeros((dvec.shape[0], nmp, AngleIndex.N_ANGLES))
        out[..., AngleIndex.AOA_PHI]    = nlos_aoa_phi
        out[..., AngleIndex.AOA_THETA]  = nlos_aoa_theta
        out[..., AngleIndex.AOD_PHI]    = nlos_aod_phi
        out[..., AngleIndex.AOD_THETA]  = nlos_aod_theta
        return out

    # --------------------------- Delays transformation --------------------------- #
    
    def _transform_delays(self,
        dvec: np.ndarray, nlos_delays: np.ndarray, fit: bool = False
    ) -> np.ndarray:
        """
        """
        # LOS delays distance through speed (distance / c)
        distance = np.linalg.norm(dvec, axis=1)
        los_delays = distance / LIGHT_SPEED

        # Relative excess delay >= 0
        relative = np.maximum(0.0, nlos_delays - los_delays[:, None])
        if fit: self.delay_scale = np.mean(relative)
        return relative / self.delay_scale
    
    
    def _inverse_transform_delays(self,
        dvec: np.ndarray, delays: np.ndarray
    ) -> np.ndarray:
        """
        """
        # Compute LOS delays
        distance = np.linalg.norm(dvec, axis=1)
        los_delays = distance / LIGHT_SPEED
        
        # return the computed absolute delays
        return delays * self.delay_scale + los_delays[:, None]
