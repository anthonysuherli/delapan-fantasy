"""
GPU Batch Training System for XGBoost Models.

This module provides efficient batch training of multiple XGBoost models
on GPU hardware, maximizing GPU utilization and reducing training time.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

logger = logging.getLogger(__name__)


class GPUBatchTrainer:
    """
    Efficient batch training system for GPU XGBoost models.
    
    Manages GPU memory, coordinates parallel training, and optimizes
    resource utilization across multiple models.
    """
    
    def __init__(
        self,
        batch_size: int = 8,
        gpu_device: str = "cuda:0",
        max_workers: int = 4,
        memory_limit_mb: Optional[int] = None
    ):
        """
        Initialize GPU batch trainer.
        
        Parameters
        ----------
        batch_size : int, optional
            Number of models to train simultaneously. Default: 8
        gpu_device : str, optional  
            GPU device specification. Default: "cuda:0"
        max_workers : int, optional
            Maximum concurrent training threads. Default: 4
        memory_limit_mb : Optional[int]
            GPU memory limit in MB. If None, uses available memory.
        """
        self.batch_size = batch_size
        self.gpu_device = gpu_device
        self.max_workers = max_workers
        self.memory_limit_mb = memory_limit_mb
        
        # Training statistics
        self.stats = {
            'models_trained': 0,
            'total_training_time': 0.0,
            'batch_count': 0,
            'average_batch_time': 0.0,
            'gpu_memory_peak_mb': 0.0
        }
        
        # Thread safety
        self._lock = threading.Lock()
        
        logger.info(f"Initialized GPUBatchTrainer: batch_size={batch_size}, device={gpu_device}")
    
    def train_batch(
        self,
        training_batch: List[Dict[str, Any]],
        model_params: Dict[str, Any],
        save_models: bool = True,
        models_dir: Optional[Path] = None
    ) -> List[Dict[str, Any]]:
        """
        Train a batch of models efficiently on GPU.
        
        Parameters
        ----------
        training_batch : List[Dict[str, Any]]
            List of training data dictionaries, each containing:
            - 'player_id': str
            - 'player_name': str  
            - 'X_train': pd.DataFrame or np.ndarray
            - 'y_train': pd.Series or np.ndarray
            - 'metadata': dict (optional)
        model_params : Dict[str, Any]
            XGBoost model parameters
        save_models : bool, optional
            Whether to save trained models. Default: True
        models_dir : Optional[Path]
            Directory to save models
            
        Returns
        -------
        List[Dict[str, Any]]
            List of training results with models and metadata
        """
        if not training_batch:
            return []
            
        batch_start_time = time.perf_counter()
        batch_size = len(training_batch)
        
        logger.info(f"Training batch of {batch_size} models on {self.gpu_device}")
        
        # Optimize batch for GPU memory
        optimized_batch = self._optimize_batch_for_gpu(training_batch)
        
        # Train models in parallel with GPU coordination  
        results = []
        if batch_size <= self.max_workers:
            # Small batch - use threading for coordination
            results = self._train_batch_threaded(optimized_batch, model_params, save_models, models_dir)
        else:
            # Large batch - process in sub-batches
            results = self._train_batch_chunked(optimized_batch, model_params, save_models, models_dir)
        
        batch_time = time.perf_counter() - batch_start_time
        
        # Update statistics
        with self._lock:
            self.stats['models_trained'] += len(results)
            self.stats['total_training_time'] += batch_time
            self.stats['batch_count'] += 1
            self.stats['average_batch_time'] = self.stats['total_training_time'] / self.stats['batch_count']
        
        logger.info(f"Batch training completed: {len(results)} models in {batch_time:.2f}s "
                   f"({len(results)/batch_time:.2f} models/sec)")
        
        return results
    
    def _optimize_batch_for_gpu(self, training_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Optimize training batch for GPU memory efficiency.
        
        Parameters
        ----------
        training_batch : List[Dict[str, Any]]
            Original training batch
            
        Returns
        -------
        List[Dict[str, Any]]
            Optimized batch with GPU-friendly data formats
        """
        optimized = []
        
        for item in training_batch:
            optimized_item = item.copy()
            
            # Convert to GPU-optimal data types
            X = item['X_train']
            y = item['y_train']
            
            # Convert pandas to numpy for GPU efficiency
            if isinstance(X, pd.DataFrame):
                X = X.values.astype(np.float32, copy=False)
            elif isinstance(X, np.ndarray):
                X = X.astype(np.float32, copy=False)
                
            if isinstance(y, pd.Series):
                y = y.values.astype(np.float32, copy=False)
            elif isinstance(y, np.ndarray):
                y = y.astype(np.float32, copy=False)
            
            # Ensure C-contiguous memory layout
            X = np.ascontiguousarray(X)
            y = np.ascontiguousarray(y)
            
            optimized_item['X_train'] = X
            optimized_item['y_train'] = y
            
            optimized.append(optimized_item)
        
        return optimized
    
    def _train_batch_threaded(
        self, 
        training_batch: List[Dict[str, Any]], 
        model_params: Dict[str, Any],
        save_models: bool,
        models_dir: Optional[Path]
    ) -> List[Dict[str, Any]]:
        """
        Train batch using ThreadPoolExecutor for GPU coordination.
        """
        results = []
        
        # Ensure GPU device is set in model params
        gpu_params = model_params.copy()
        gpu_params['device'] = self.gpu_device
        gpu_params['tree_method'] = 'hist'  # Optimal for GPU
        
        with ThreadPoolExecutor(max_workers=min(len(training_batch), self.max_workers)) as executor:
            # Submit all training jobs
            future_to_player = {}
            
            logger.info(f"Submitting {len(training_batch)} training jobs to ThreadPoolExecutor")
            
            for item in training_batch:
                try:
                    future = executor.submit(
                        self._train_single_model_gpu,
                        item, 
                        gpu_params, 
                        save_models, 
                        models_dir
                    )
                    future_to_player[future] = item['player_id']
                    logger.debug(f"Submitted training job for player {item['player_id']}")
                except Exception as e:
                    logger.error(f"Failed to submit training job for player {item['player_id']}: {str(e)}")
            
            logger.info(f"All {len(future_to_player)} jobs submitted, waiting for completion...")
            
            # Collect results as they complete with timeout
            completed_count = 0
            try:
                for future in as_completed(future_to_player, timeout=300):  # 5 minute timeout
                    player_id = future_to_player[future]
                    completed_count += 1
                    
                    try:
                        logger.debug(f"Processing result for player {player_id} ({completed_count}/{len(future_to_player)})")
                        result = future.result(timeout=60)  # 1 minute timeout per result
                        
                        if result is not None:
                            results.append(result)
                            logger.debug(f"Successfully processed player {player_id}")
                        else:
                            logger.warning(f"No result returned for player {player_id}")
                            
                    except Exception as e:
                        logger.error(f"Training failed for player {player_id}: {str(e)}")
                        import traceback
                        logger.error(f"Full traceback: {traceback.format_exc()}")
                        
            except Exception as timeout_error:
                logger.error(f"Batch training timed out after 300 seconds: {str(timeout_error)}")
                logger.error(f"Completed {completed_count}/{len(future_to_player)} models before timeout")
                
                # Cancel remaining futures
                for future in future_to_player:
                    if not future.done():
                        future.cancel()
                        logger.warning(f"Cancelled training for player {future_to_player[future]}")
        
        return results
    
    def _train_batch_chunked(
        self,
        training_batch: List[Dict[str, Any]], 
        model_params: Dict[str, Any],
        save_models: bool,
        models_dir: Optional[Path]
    ) -> List[Dict[str, Any]]:
        """
        Train large batch in smaller chunks to manage GPU memory.
        """
        results = []
        chunk_size = self.max_workers
        
        for i in range(0, len(training_batch), chunk_size):
            chunk = training_batch[i:i + chunk_size]
            chunk_results = self._train_batch_threaded(chunk, model_params, save_models, models_dir)
            results.extend(chunk_results)
            
            # Optional: Clear GPU memory between chunks
            self._clear_gpu_memory()
        
        return results
    
    def _train_single_model_gpu(
        self,
        training_item: Dict[str, Any],
        model_params: Dict[str, Any],
        save_models: bool,
        models_dir: Optional[Path]
    ) -> Optional[Dict[str, Any]]:
        """
        Train a single model on GPU with optimized parameters.
        """
        player_id = training_item.get('player_id', 'unknown')
        player_name = training_item.get('player_name', 'unknown')
        
        try:
            logger.debug(f"Starting GPU training for {player_name} ({player_id})")
            
            # Import XGBoost
            try:
                import xgboost as xgb
                logger.debug(f"XGBoost version: {xgb.__version__}")
            except ImportError as e:
                logger.error(f"XGBoost import failed: {str(e)}")
                return None
            
            X_train = training_item['X_train']
            y_train = training_item['y_train']
            
            logger.debug(f"Training data shapes - X: {X_train.shape}, y: {y_train.shape}")
            
            # Validate input data
            if len(X_train) < 3 or len(y_train) < 3:
                logger.debug(f"Insufficient data for {player_name}: {len(X_train)} samples")
                return None
            
            logger.debug(f"Model parameters for {player_name}: {model_params}")
            
            # Create and train XGBoost model
            try:
                model = xgb.XGBRegressor(**model_params)
                logger.debug(f"Created XGBRegressor for {player_name}")
            except Exception as e:
                logger.error(f"Failed to create XGBRegressor for {player_name}: {str(e)}")
                return None
            
            train_start = time.perf_counter()
            logger.debug(f"Starting model.fit() for {player_name}...")
            
            try:
                model.fit(X_train, y_train)
                train_time = time.perf_counter() - train_start
                logger.debug(f"Model training completed for {player_name} in {train_time:.2f}s")
            except Exception as e:
                logger.error(f"Model training failed for {player_name}: {str(e)}")
                return None
            
            # Generate prediction on latest features for projection
            latest_features = X_train[-1:] if len(X_train.shape) == 2 else X_train[-1].reshape(1, -1)
            
            # Handle device-consistent predictions to avoid XGBoost warnings
            prediction = self._predict_device_consistent(model, latest_features, model_params)
            
            result = {
                'player_id': player_id,
                'player_name': player_name,
                'model': model,
                'prediction': prediction,
                'training_samples': len(X_train),
                'training_time': train_time,
                'metadata': training_item.get('metadata', {})
            }
            
            # Save model if requested
            if save_models and models_dir:
                self._save_gpu_model(model, player_id, player_name, models_dir, result)
            
            return result
            
        except Exception as e:
            logger.error(f"GPU training failed for {training_item.get('player_name', 'unknown')}: {str(e)}")
            return None
    
    def _save_gpu_model(
        self, 
        model, 
        player_id: str, 
        player_name: str, 
        models_dir: Path,
        result: Dict[str, Any]
    ):
        """
        Save GPU-trained model to disk with metadata.
        """
        try:
            models_dir = Path(models_dir)
            models_dir.mkdir(parents=True, exist_ok=True)
            
            # Create safe filename
            safe_name = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in player_name)
            safe_name = safe_name.replace(' ', '_')
            
            model_file = models_dir / f"{safe_name}_{player_id}_gpu.json"
            
            # Save in XGBoost native format for GPU compatibility
            model.save_model(str(model_file))
            
            # Save metadata
            metadata_file = model_file.with_suffix('.json')
            import json
            
            metadata = {
                'player_id': player_id,
                'player_name': player_name,
                'training_samples': result['training_samples'],
                'training_time': result['training_time'],
                'model_file': model_file.name,
                'gpu_device': self.gpu_device,
                'trained_on_gpu': True
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            logger.debug(f"Saved GPU model for {player_name}: {model_file}")
            
        except Exception as e:
            logger.warning(f"Failed to save model for {player_name}: {str(e)}")
    
    def _predict_device_consistent(
        self,
        model,
        features: np.ndarray,
        model_params: Dict[str, Any]
    ) -> float:
        """
        Make prediction with device consistency to avoid XGBoost warnings.
        
        Parameters
        ----------
        model : XGBRegressor
            Trained XGBoost model
        features : np.ndarray
            Feature array for prediction
        model_params : Dict[str, Any]
            Model parameters containing device info
            
        Returns
        -------
        float
            Prediction value
        """
        try:
            device = model_params.get('device', 'cpu')
            
            if 'cuda' in str(device):
                # For GPU models, use the approach that avoids device mismatch warnings
                import xgboost as xgb
                import warnings
                
                # Suppress the specific device mismatch warning during prediction
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='.*Falling back to prediction using DMatrix.*')
                    
                    # Set booster device before prediction
                    booster = model.get_booster()
                    booster.set_param('device', device)
                    
                    # Use DMatrix for GPU-compatible prediction
                    dtest = xgb.DMatrix(features)
                    predictions = booster.predict(dtest)
                    return float(predictions[0])
            else:
                # For CPU models, standard prediction is fine
                return float(model.predict(features)[0])
                
        except Exception as e:
            logger.debug(f"Device-consistent prediction failed, using fallback: {str(e)}")
            # Fallback to standard prediction
            try:
                return float(model.predict(features)[0])
            except Exception as fallback_error:
                logger.warning(f"All prediction methods failed for model: {str(fallback_error)}")
                return 0.0  # Return safe default
    
    def _clear_gpu_memory(self):
        """
        Clear GPU memory to prevent memory leaks.
        """
        try:
            import cupy as cp
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
            logger.debug("Cleared GPU memory pool")
        except ImportError:
            # CuPy not available, rely on XGBoost memory management
            pass
        except Exception as e:
            logger.debug(f"GPU memory cleanup failed: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get training statistics.
        
        Returns
        -------
        Dict[str, Any]
            Training performance statistics
        """
        with self._lock:
            return self.stats.copy()
    
    def reset_stats(self):
        """
        Reset training statistics.
        """
        with self._lock:
            self.stats = {
                'models_trained': 0,
                'total_training_time': 0.0,
                'batch_count': 0,
                'average_batch_time': 0.0,
                'gpu_memory_peak_mb': 0.0
            }
            
    def __del__(self):
        """
        Cleanup on destruction.
        """
        try:
            self._clear_gpu_memory()
        except:
            pass