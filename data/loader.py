"""
    data/loader.py
    --------------
    This script contains modules and functions for managing the 
    data loading for the UAV model. This includes structures of 
    shuffle and splitting data, a `DataLoader` module, which 
    load/save data.
    
    dvec        vector          (N,3):  np.float32
    rx_type     scalar          (N,):   np.uint8
    link_state  scalar          (N,):   np.uint8
    los_pl      scalar          (N,)    np.float32
    los_ang     vector          (N,2)   np.float32
    los_dly     scalar          (N,)    np.float32
    nlos_pl     list of floats  (N,)    np.ndarray(dtype=object)
    nlos_ang    list of vectors (N,)    np.ndarray(dtype=object)
    nlos_dly    list of floats  (N,)    np.ndarray(dtype=object)
"""
import json
import orjson
import logging

import multiprocessing as mp
import pandas as pd
import numpy as np

from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple, Union
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor



def shuffle_and_split(
    data        : Dict[str, np.ndarray],
    val_ratio   : float = 0.20,
    seed        : int   = 42,
    n_workers   : Optional[int] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
        Shuffles the dataset passed into a divded dataset, into
        `training set` and `validation set`.
        
        Args:
        -----
            data: Dictionary of arrays (all must have same length)
            val_ratio: Ratio of samples to use for validation.
            seed:  Random seed for reproducibility
            n_workers:  If > 1, use ThreadPoolExecutor to parallelize 
                        splits.
        Returns:
            Tuple of two dictionaries representing the training dataset
            and the validation dataset.
    """
    # --- Sanity check: Guarantees that all arrays have same lengths --- #
    lengths = [len(value) for value in data.values()]
    if len(set(lengths)) != 1:
        raise ValueError(f"Inconsistent array lengths: {lengths}")
    n = lengths[0]

    # ----- Shuffle the indices ----- #
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)

    split_index = int(n * (1 - val_ratio))
    train_idx, val_idx = indices[:split_index], indices[split_index:]

    # ----- Splitting function ----- #
    def split_array(key: str):
        arr = data[key]
        return key, arr[train_idx], arr[val_idx]
    
    # ----- Parallel or Sequential Processing ----- #
    if n_workers is None or n_workers <= 1:
        results = [split_array(key) for key in data.keys()]
    else:
        with ThreadPoolExecutor(max_workers=min(n_workers, mp.cpu_count())) as executor:
            results = list(executor.map(split_array, data.keys()))
    
    return ({k: train for k, train, _ in results},
            {k: val for k, _, val in results},)




class DataLoader:
    """
    """
    
    REQUIRED_COLUMNS = ['dvec','rx_type','link_state','los_pl', 
                        'los_ang','los_dly','nlos_pl','nlos_ang','nlos_dly']
    LIST_COLUMNS = ['dvec','los_ang','nlos_pl','nlos_ang','nlos_dly']
    SCHEMA = {"dvec"        : (np.float32,  True ),
              "rx_type"     : (np.uint8,    False),
              "link_state"  : (np.uint8,    False),
              "los_pl"      : (np.float32,  False),
              "los_ang"     : (np.float32,  True ),
              "los_dly"     : (np.float32,  False),
              "nlos_pl"     : (np.float32,  True ),
              "nlos_ang"    : (np.float32,  True ),
              "nlos_dly"    : (np.float32,  True )}

    def __init__(self,
        n_workers   : Optional[int] = None,
        chunksize   : int   = 10_000,
        debug_level : int   = logging.ERROR
    ):
        """
            Loader of channel-data for UAVs flight paths
        """
        self.directory = Path(__file__).parent / "base"
        self.directory.mkdir(parents=True, exist_ok=True)

        self.n_workers = mp.cpu_count() if n_workers is None else n_workers
        self.chunksize = int(chunksize)
        self.parallel = True if n_workers != 1 else False

        self._setup_logging(level=debug_level)
    

    # =============================================================== #
    # ---------------========== Save / Load ==========--------------- #
    # =============================================================== #

    def save(self,
        data        : Dict[str, np.ndarray],
        filepath    : Union[str, Path],
        fmt         : str = "csv"
    ) -> None:
        """
            Save the full processed data directory (NumPy arrays) 
            to disk, storage memory. This is used mainly for pass
            data composed of the same internal structure for the
            UAV ChannelData.

            Args:
            -----
                data:   Dictionary of string name and NumPy Arrays
                filepath:   Destination filepath where the stored 
                            data is to be stored.
                format: File-format the data is being stored as

            Raises:
            -------
                ValueError: If format the file is being saved is
                            not supported, and value error is raised.
        """
        fmt = fmt.lower() # to lower just in case, does it once, hefty op
        
        filepath = Path(self.directory) / filepath
        filepath = filepath.with_suffix(f".{fmt}")
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Convert NumPy arrays into lists
        df = pd.DataFrame({key: [
            v.tolist() if isinstance(v, np.ndarray) else v for v in arr
        ] for key, arr in data.items()})

        
        if fmt == "csv": df.to_csv(filepath, index=False)
        elif fmt == "json": df.to_json(filepath, orient="records", lines=True)
        elif fmt == "parquet": df.to_parquet(filepath, index=False)
        else: raise ValueError(f"Unsupported file-format: {fmt}")

        self.logger.info(f"saved `{len(df)}` rows to `{filepath}`")
    

    def load(self,
        filepaths: Union[str, List[str]]
    ) -> Dict[str, np.ndarray]:
        """
            Loading the UAV flight path-data from one or more files
            into a dictionary consisting of the string representation
            accomplished with corresponding np.ndarray data.

            This method takes eitehr a string or a list of file paths
            relative to the base directory 'self.dir'. If multiple
            files are passed, these files are concatenated into a 
            single result file before return it. This method also 
            enable and support parallel computation based on the 
            value of 'self.parallel' for faster computations to 
            improve efficiency in concatenating the files.

            Args:
            -----
                filepaths:  Path or list of paths corresponding to 
                            the files to load.
            Returns:
            --------
                Dict[ str, np.ndarray ]:    Dictionary of concatenated
                                            NumPy arrays
            Raises:
            -------
                Exception:  If loading or trying to processing any of 
                            the files would fail.
            Examples:
            ---------
                >>> Loader  = DataLoader()
                >>> data    = loader.load([ "f1.csv", "f2.csv" ])
            ---------
            will load the files 'f1.csv' and 'f2.csv' into a concatenated
            file containing all information from both files stored in
            the placeholder variable 'data'.\\
        """
        try:
            filepaths = [filepaths] if isinstance(filepaths, str) else filepaths
            self.logger.info(f"Loading `{len(filepaths)}` file(s) using"
                             f" {self.n_workers} workers...")

            chunks: List[Dict[str, np.ndarray]] = []
            for filepath in filepaths:
                path = Path(self.directory) / filepath
                try: chunks.extend(list(self._load_file_chunks(path)))
                except Exception as e:
                    self.logger.error(f"Failed to load `{filepaths}`: {e}")
                    raise
            
            if not chunks: raise RuntimeError("No data chunks were loaded")
            
            exe = ProcessPoolExecutor if self.parallel else ThreadPoolExecutor
            try:
                with exe(max_workers=self.n_workers) as executor:
                    results = list(executor.map(self._process_chunk, chunks))
            except Exception as e:
                self.logger.error(f"Parallel processing failed: {e}")
                self.logger.warning("Falling back to sequential processing...")
                results = [self._process_chunk(chunk) for chunk in chunks]
            return self._concatenate_results(results)
        
        except Exception as e:
            self.logger.exception(f"Failed to load data: {e}")
            raise
    


    # ================================================================ #
    # ---------------======== Internal Methods ========--------------- #
    # ================================================================ #

    def _setup_logging(self, level: int = logging.DEBUG):
        """
            Setting up the logger used to provide feedback to the 
            data loader as a response to operations that is being 
            conductedby the loader.

            Examples:
            ---------
                >>> import logging
                >>> from data.loader import DataLoader
                >>> loader = DataLoader()
                >>> loader._setup_logging(debugging=logging.DEBUG)
                >>> loader._setup_logging(debugging=logging.INFO)
                >>> loader._setup_logging(debugging=logging.WARNING)
                >>> loader._setup_logging(debugging=logging.ERROR)
                >>> loader._setup_logging(debugging=logging.CRITICAL)
            ---------
            Internal methods called by the constructor to initialize the 
            logger, logging is mainly useful for debugging level. It 
            also support various 'LogLevel' to more flexibility to which
            feedback is aimed to return a feedback based on a particular 
            specific concern.\\
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            self.logger.addHandler(handler)

        self.logger.setLevel(level=level)
        self.logger.propagate = False
    

    def _load_file_chunks(self, filepath: Path) -> Generator[pd.DataFrame, None, None]:
        """
            Generator that yields chunks to the assignment file
            for more efficient processing of the file.

            Args:
            -----
                filepath:   Path to the input data file to be processed.

            Yields:
            ------
                pd.DataFrame:   Chunks of processed data files as
                                pandas data-frames
            Raises:
            -------
                FileNotFoundError:  If the file does not exist, the 
                                    method raises an error unable to
                                    find the requested file passed.
                ValueError: If the file format is unsupported or for
                            any reason unchunkable and value-error 
                            is raised.
            -------
        """
        if not filepath.exists():
            raise FileExistsError(f"File `{filepath}` not found")
        
        if filepath.suffix == ".csv":
            yield from pd.read_csv(filepath, chunksize=self.chunksize)
        elif filepath.suffix == ".json":
            yield from pd.read_json(filepath, lines=True, chunksize=self.chunksize)
        elif filepath.suffix == ".parquet":
            df = pd.read_parquet(filepath)
            for i in range(0, len(df), self.chunksize):
                yield df.iloc[i: i + self.chunksize]
        else:
            raise ValueError(f"Unsupported format: {filepath.suffix}")


    def _process_chunk(self, chunk: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
            Process a single chunk into structured NumPy arrays.
            
            Steps:
            1. Convert stringified lists into Python lists (fast orjson).
            2. Convert pandas DataFrame columns to NumPy arrays with schema dtypes.
        """
        # 1. Parse JSON-encoded list columns if needed
        for column in self.LIST_COLUMNS:
            if chunk[column].dtype == object and isinstance(chunk[column].iloc[0], str):
                try: chunk[column] = chunk[column].map(orjson.loads)
                except Exception as e:
                    self.logger.error(f"Failed to parse JSON in column `{column}`: {e}")
                    raise
        
        # 2. Convert to NumPy arrays with correct dtype and shape
        processed: Dict[str, np.ndarray] = {}
        for column, (dtype, need_stack) in self.SCHEMA.items():
            values = chunk[column].to_numpy()
            if need_stack:
                try: values = np.stack(values)
                except Exception as e:
                    self.logger.error(f"Failed to stack column `{column}`: {e}")
                    raise
            
            processed[column] = values.astype(dtype, copy=False)
        return processed
    

    def _concatenate_results(self, 
        results : List[Dict[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """
            Concatenates a list of dictionaries (processed chunks) into
            a single dataset dictionary.

            Args:
            -----
                results:    List of processed chunk dictionaries passed to
                            be componsed into a dataset.
            Returns:
            --------
                Dict[ str, np.ndarray ]:    Single concatenated dictionary of 
                                            the list of passed dictionaries.
        """
        # If empty managing returning an empty dictionary of empty arrays
        if not results:
            self.logger.warning("No results to concatenate - returning empty arrays")
            return {key: np.empty((0,)) for key in self.REQUIRED_COLUMNS}

        concatenated: Dict[str, np.ndarray] = {}
        keys = results[0].keys()
        for key in keys:
            try:
                arrays = [result[key] for result in results]
                if not arrays:
                    raise ValueError(f"No data found for key '{key}' in results")

                sample = arrays[0]
                concat_fn = np.vstack if sample.ndim > 1 else np.concatenate
                concatenated[key] = concat_fn(arrays)
                self.logger.debug(f"Concatenated '{key}' to shape "
                                  f"{concatenated[key].shape}")
            except Exception as e:
                self.logger.error(f"Error concatenated key '{key}': {e}")
                raise

        return concatenated