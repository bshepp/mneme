"""Symbolic regression for discovering field dynamics equations.

This module provides a SymbolicRegressor that wraps PySR (when available)
to discover interpretable mathematical equations governing field evolution.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Union
import numpy as np

# Try to import PySR
_PYSR_AVAILABLE = False
_PySRRegressor = None

try:
    from pysr import PySRRegressor as _PySRRegressor
    _PYSR_AVAILABLE = True
except ImportError:
    pass


class SymbolicRegressor:
    """Discover symbolic equations from field dynamics.
    
    This class wraps PySR to find interpretable mathematical equations
    that describe how field values evolve over time or relate to each other.
    
    When PySR is not installed, it operates in placeholder mode and returns
    simple fallback equations.
    
    Parameters
    ----------
    operators : List[str], optional
        Mathematical operators to use. Default includes basic arithmetic
        and trigonometric functions.
    complexity_penalty : float
        Penalty for equation complexity (higher = simpler equations).
        Maps to PySR's parsimony parameter.
    max_complexity : int
        Maximum allowed complexity for equations.
    populations : int
        Number of populations for genetic algorithm.
    niterations : int
        Number of iterations for symbolic search.
    binary_operators : List[str], optional
        Binary operators (e.g., +, -, *, /). If None, derived from operators.
    unary_operators : List[str], optional
        Unary operators (e.g., sin, cos, exp). If None, derived from operators.
    extra_sympy_mappings : Dict[str, Any], optional
        Additional SymPy function mappings.
    loss : str
        Loss function for PySR. Default is 'loss(prediction, target) = (prediction - target)^2'.
    denoise : bool
        Whether to denoise data before fitting.
    select_k_features : int, optional
        If set, use only the k most important features.
    progress : bool
        Whether to show progress during fitting.
    
    Examples
    --------
    >>> import numpy as np
    >>> from mneme.models.symbolic import SymbolicRegressor
    >>> 
    >>> # Generate sample dynamics data
    >>> t = np.linspace(0, 10, 100)
    >>> X = np.column_stack([t, np.sin(t)])
    >>> y = 2.0 * np.sin(t) + 0.5 * t  # True relationship
    >>> 
    >>> # Fit symbolic regressor
    >>> sr = SymbolicRegressor(niterations=40)
    >>> sr.fit(X, y, variable_names=['t', 'sin_t'])
    >>> 
    >>> # Get discovered equations
    >>> equations = sr.get_equations()
    >>> print(equations[0])  # Best equation
    """
    
    # Default operators for field dynamics
    DEFAULT_BINARY_OPS = ['+', '-', '*', '/', 'pow']
    DEFAULT_UNARY_OPS = ['sin', 'cos', 'exp', 'log', 'sqrt', 'abs', 'neg']
    
    def __init__(
        self,
        operators: Optional[List[str]] = None,
        complexity_penalty: float = 0.001,
        max_complexity: int = 30,
        populations: int = 15,
        niterations: int = 100,
        binary_operators: Optional[List[str]] = None,
        unary_operators: Optional[List[str]] = None,
        extra_sympy_mappings: Optional[Dict[str, Any]] = None,
        loss: str = "L2DistLoss()",
        denoise: bool = False,
        select_k_features: Optional[int] = None,
        progress: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        self.operators = operators
        self.complexity_penalty = complexity_penalty
        self.max_complexity = max_complexity
        self.populations = populations
        self.niterations = niterations
        self.loss = loss
        self.denoise = denoise
        self.select_k_features = select_k_features
        self.progress = progress
        self.random_state = random_state
        self.extra_sympy_mappings = extra_sympy_mappings or {}
        
        # Parse operators into binary and unary
        if binary_operators is not None:
            self.binary_operators = binary_operators
        elif operators is not None:
            self.binary_operators = [op for op in operators if op in ['+', '-', '*', '/', 'pow', '^', '**']]
        else:
            self.binary_operators = self.DEFAULT_BINARY_OPS
            
        if unary_operators is not None:
            self.unary_operators = unary_operators
        elif operators is not None:
            self.unary_operators = [op for op in operators if op in ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'abs', 'neg', 'square', 'cube']]
        else:
            self.unary_operators = self.DEFAULT_UNARY_OPS
        
        # Internal state
        self._model: Optional[Any] = None
        self._equations: List[str] = []
        self._equation_scores: List[float] = []
        self._variable_names: Optional[List[str]] = None
        self._is_fitted: bool = False
        self._using_pysr: bool = False
        
    @property
    def is_available(self) -> bool:
        """Check if PySR is available."""
        return _PYSR_AVAILABLE
    
    def fit(
        self,
        X: Union[np.ndarray, List],
        y: Union[np.ndarray, List],
        variable_names: Optional[List[str]] = None,
        weights: Optional[np.ndarray] = None,
    ) -> 'SymbolicRegressor':
        """Fit symbolic regressor to discover equations.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features (e.g., spatial coordinates, time, field values).
        y : array-like of shape (n_samples,)
            Target values to predict (e.g., field evolution, gradients).
        variable_names : List[str], optional
            Names for input variables. If None, uses x0, x1, etc.
        weights : np.ndarray, optional
            Sample weights for weighted regression.
            
        Returns
        -------
        self : SymbolicRegressor
            Fitted regressor.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_features = X.shape[1]
        self._variable_names = variable_names or [f"x{i}" for i in range(n_features)]
        
        if _PYSR_AVAILABLE and _PySRRegressor is not None:
            self._fit_pysr(X, y, weights)
            self._using_pysr = True
        else:
            self._fit_fallback(X, y)
            self._using_pysr = False
            warnings.warn(
                "PySR not available. Using fallback placeholder. "
                "Install with: pip install pysr",
                UserWarning
            )
        
        self._is_fitted = True
        return self
    
    def _fit_pysr(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> None:
        """Fit using PySR."""
        # Create PySR model with our configuration
        self._model = _PySRRegressor(
            binary_operators=self.binary_operators,
            unary_operators=self.unary_operators,
            niterations=self.niterations,
            populations=self.populations,
            maxsize=self.max_complexity,
            parsimony=self.complexity_penalty,
            loss=self.loss,
            denoise=self.denoise,
            select_k_features=self.select_k_features,
            progress=self.progress,
            random_state=self.random_state,
            extra_sympy_mappings=self.extra_sympy_mappings if self.extra_sympy_mappings else None,
            temp_equation_file=True,  # Don't leave temp files
            verbosity=1 if self.progress else 0,
        )
        
        # Fit the model
        self._model.fit(X, y, variable_names=self._variable_names, weights=weights)
        
        # Extract equations from the Pareto front
        self._extract_equations_from_pysr()
    
    def _extract_equations_from_pysr(self) -> None:
        """Extract equations from fitted PySR model."""
        if self._model is None:
            return
            
        try:
            # Get the equations DataFrame
            equations_df = self._model.equations_
            
            if equations_df is not None and len(equations_df) > 0:
                # Sort by score (lower is better) and extract
                sorted_eqs = equations_df.sort_values('loss')
                
                self._equations = []
                self._equation_scores = []
                
                for _, row in sorted_eqs.iterrows():
                    # Get sympy equation string
                    eq_str = str(row.get('equation', row.get('sympy_format', '')))
                    score = float(row.get('loss', row.get('score', float('inf'))))
                    
                    self._equations.append(eq_str)
                    self._equation_scores.append(score)
        except Exception as e:
            warnings.warn(f"Could not extract equations from PySR: {e}")
            self._equations = ["<extraction_failed>"]
            self._equation_scores = [float('inf')]
    
    def _fit_fallback(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fallback fitting when PySR is not available.
        
        Uses simple linear regression to provide a basic equation.
        """
        from sklearn.linear_model import LinearRegression
        
        # Fit simple linear model as fallback
        lr = LinearRegression()
        lr.fit(X, y)
        
        # Build equation string
        terms = []
        for i, (name, coef) in enumerate(zip(self._variable_names or [], lr.coef_)):
            if abs(coef) > 1e-10:
                terms.append(f"{coef:.4f}*{name}")
        
        if abs(lr.intercept_) > 1e-10:
            terms.append(f"{lr.intercept_:.4f}")
        
        if terms:
            eq_str = " + ".join(terms)
        else:
            eq_str = "0"
        
        self._equations = [eq_str]
        self._equation_scores = [float(np.mean((lr.predict(X) - y) ** 2))]
        self._model = lr
    
    def predict(self, X: Union[np.ndarray, List]) -> np.ndarray:
        """Predict using the best discovered equation.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.
            
        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predictions.
        """
        if not self._is_fitted:
            raise RuntimeError("SymbolicRegressor must be fitted before prediction")
        
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        if self._model is not None:
            try:
                return self._model.predict(X)
            except Exception as e:
                warnings.warn(f"Prediction failed: {e}")
                return np.zeros(len(X))
        
        return np.zeros(len(X))
    
    def get_equations(self, n_best: Optional[int] = None) -> List[str]:
        """Get discovered equations.
        
        Parameters
        ----------
        n_best : int, optional
            Return only the n best equations. If None, returns all.
            
        Returns
        -------
        equations : List[str]
            List of equation strings, sorted by score (best first).
        """
        if n_best is None:
            return list(self._equations)
        return list(self._equations[:n_best])
    
    def get_best_equation(self) -> str:
        """Get the single best equation.
        
        Returns
        -------
        equation : str
            Best equation string.
        """
        if self._equations:
            return self._equations[0]
        return "<no_equation_found>"
    
    def get_equation_scores(self) -> List[float]:
        """Get scores for all equations.
        
        Returns
        -------
        scores : List[float]
            Loss/score values for each equation (lower is better).
        """
        return list(self._equation_scores)
    
    def get_sympy_expression(self, equation_index: int = 0) -> Any:
        """Get SymPy expression for an equation.
        
        Parameters
        ----------
        equation_index : int
            Index of equation to get (0 = best).
            
        Returns
        -------
        expr : sympy.Expr or None
            SymPy expression, or None if not available.
        """
        if not self._using_pysr or self._model is None:
            try:
                import sympy
                return sympy.sympify(self._equations[equation_index])
            except Exception:
                return None
        
        try:
            return self._model.sympy(equation_index)
        except Exception:
            return None
    
    def latex(self, equation_index: int = 0) -> str:
        """Get LaTeX representation of an equation.
        
        Parameters
        ----------
        equation_index : int
            Index of equation to render (0 = best).
            
        Returns
        -------
        latex_str : str
            LaTeX string for the equation.
        """
        expr = self.get_sympy_expression(equation_index)
        if expr is not None:
            try:
                import sympy
                return sympy.latex(expr)
            except Exception:
                pass
        
        # Fallback: return raw equation
        if equation_index < len(self._equations):
            return self._equations[equation_index]
        return ""
    
    def score(self, X: Union[np.ndarray, List], y: Union[np.ndarray, List]) -> float:
        """Compute R² score on test data.
        
        Parameters
        ----------
        X : array-like
            Test features.
        y : array-like
            True target values.
            
        Returns
        -------
        r2 : float
            R² score (1.0 = perfect, 0.0 = baseline).
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        y_pred = self.predict(X)
        
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        
        return 1.0 - (ss_res / ss_tot)


def discover_field_dynamics(
    field_sequence: np.ndarray,
    dt: float = 1.0,
    spatial_features: bool = True,
    niterations: int = 100,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Discover symbolic equations governing field evolution.
    
    This is a convenience function that extracts relevant features from
    a field time series and runs symbolic regression to find governing equations.
    
    Parameters
    ----------
    field_sequence : np.ndarray
        Field evolution with shape (timesteps, height, width).
    dt : float
        Time step between frames.
    spatial_features : bool
        Whether to include spatial derivative features.
    niterations : int
        Number of PySR iterations.
    **kwargs
        Additional arguments passed to SymbolicRegressor.
        
    Returns
    -------
    result : Dict[str, Any]
        Dictionary containing:
        - 'equations': List of discovered equations
        - 'best_equation': Best equation string
        - 'regressor': Fitted SymbolicRegressor
        - 'features_used': Names of input features
        - 'r2_score': R² score on training data
    """
    if field_sequence.ndim != 3:
        raise ValueError("field_sequence must be 3D (timesteps, height, width)")
    
    T, H, W = field_sequence.shape
    
    # Extract features and targets
    features = []
    targets = []
    feature_names = ['u']  # Field value
    
    for t in range(T - 1):
        current = field_sequence[t]
        next_frame = field_sequence[t + 1]
        
        # Target: time derivative (du/dt)
        du_dt = (next_frame - current) / dt
        
        # Flatten for sampling (subsample for speed)
        step = max(1, min(H, W) // 32)
        for i in range(0, H, step):
            for j in range(0, W, step):
                feat = [current[i, j]]  # u
                
                if spatial_features:
                    # Add Laplacian (∇²u)
                    if 0 < i < H-1 and 0 < j < W-1:
                        laplacian = (
                            current[i+1, j] + current[i-1, j] +
                            current[i, j+1] + current[i, j-1] -
                            4 * current[i, j]
                        )
                        feat.append(laplacian)
                        
                        # Add gradients
                        du_dx = (current[i, j+1] - current[i, j-1]) / 2
                        du_dy = (current[i+1, j] - current[i-1, j]) / 2
                        feat.append(du_dx)
                        feat.append(du_dy)
                    else:
                        feat.extend([0, 0, 0])
                
                features.append(feat)
                targets.append(du_dt[i, j])
    
    if spatial_features:
        feature_names.extend(['laplacian_u', 'du_dx', 'du_dy'])
    
    X = np.array(features)
    y = np.array(targets)
    
    # Fit symbolic regressor
    sr = SymbolicRegressor(niterations=niterations, **kwargs)
    sr.fit(X, y, variable_names=feature_names)
    
    return {
        'equations': sr.get_equations(),
        'best_equation': sr.get_best_equation(),
        'regressor': sr,
        'features_used': feature_names,
        'r2_score': sr.score(X, y),
    }
