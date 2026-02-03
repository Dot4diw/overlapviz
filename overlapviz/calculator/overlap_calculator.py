# =============================================================================
# OverlapCalculator - Set Overlap Analysis Tool
# =============================================================================
# This module provides a comprehensive class for analyzing overlaps between
# multiple sets. It supports various input formats and provides detailed
# statistics about set intersections, exclusive elements, and pairwise relationships.
#
# Author: Dot4diw
# Version: 1.0
# =============================================================================

# Core imports for data manipulation and type hinting
import pandas as pd  # For DataFrame operations and result storage
from typing import List, Set, Dict, Tuple, Any, Optional, Union  # Type hints for better code clarity
from itertools import combinations  # For generating all possible set combinations


class OverlapCalculator:
    """
    Calculate all overlap situations for n sets
    
    This class provides comprehensive analysis of set overlaps, including:
    - All possible intersection combinations
    - Exclusive elements for each set
    - Pairwise overlap matrices
    - Summary statistics
    
    Supports multiple input formats:
    - dict: {'Set1': {"g1","g2","g3"}, 'Set2': {"g2","g3","g4"}}
    - list: [{"g1","g2","g3"}, {"g2","g3","g4"}, {"g3","g4","g5"}]
    - list of tuples: [('Set1', {"g1","g2","g3"}), ('Set2', {"g2","g3","g4"})]
    """
    
    # Class-level constants for configuration
    DEFAULT_MIN_SIZE = 1  # Default minimum overlap size for filtering
    
    def __init__(self, data: Union[Dict[str, Set], List[Set], List[Tuple[str, Set]]]):
        """
        Initialize the OverlapCalculator with input data
        
        Parameters
        ----------
        data : dict, list, or list of tuples
            Input set data in one of the supported formats:
            - Dictionary mapping set names to sets
            - List of sets (auto-named as Set1, Set2, etc.)
            - List of tuples: (set_name, set_elements)
            
        Raises
        ------
        ValueError
            If input data is empty or validation fails
        TypeError
            If input data format is not supported
            
        Examples
        --------
        >>> calc = OverlapCalculator({'Set1': {1,2,3}, 'Set2': {2,3,4}})
        >>> calc = OverlapCalculator([{1,2,3}, {2,3,4}])
        >>> calc = OverlapCalculator([('Set1', {1,2,3}), ('Set2', {2,3,4})])
        """
        # Initialize instance variables
        self.sets_names: List[str] = []  # List of set names for reference
        self.sets_list: List[Set] = []   # List of actual set objects
        self._results_df: Optional[pd.DataFrame] = None  # Cache for computed overlaps
        self._exclusive_results: Optional[pd.DataFrame] = None  # Cache for exclusive elements
        
        # Parse and validate input data
        self._parse_input(data)  # Convert input to standardized format
        self._validate_data()    # Ensure data integrity



class OverlapCalculator:
    """
    Calculate all overlap situations for n sets
    
    Supports multiple input formats:
    - dict: {'Set1': {g1,g2,g3}, 'Set2': {g2,g3,g4}}
    - list: [{g1,g2,g3}, {g2,g3,g4}, {g3,g4,g5}]
    - list of tuples: [('Set1', {g1,g2,g3}), ('Set2', {g2,g3,g4})]
    """
    
    def __init__(self, data: Union[Dict[str, Set], List[Set], List[Tuple[str, Set]]]):
        """
        Initialize calculator
        
        Parameters
        ----------
        data : dict, list, or list of tuples
            Input set data
        """
        self.sets_names: List[str] = []
        self.sets_list: List[Set] = []
        self._results_df: Optional[pd.DataFrame] = None
        self._exclusive_results: Optional[pd.DataFrame] = None  # Exclusive elements result
        
        # Parse input data
        self._parse_input(data)
        
        # Validate data
        self._validate_data()
    
    def _parse_input(self, data: Union[Dict[str, Set], List[Set], List[Tuple[str, Set]]]) -> None:
        """
        Parse input data from various supported formats into standardized internal representation
        
        This method handles three input formats and converts them to:
        - self.sets_names: List of set names (strings)
        - self.sets_list: List of set objects (Python sets)
        
        Parameters
        ----------
        data : Union[Dict[str, Set], List[Set], List[Tuple[str, Set]]]
            Input data in one of the supported formats
            
        Raises
        ------
        ValueError
            If input list is empty
        TypeError
            If input format is not supported
        """
        # Handle dictionary input: {name: set, ...}
        if isinstance(data, dict):
            self.sets_names = list(data.keys())      # Extract set names from keys
            self.sets_list = [set(v) for v in data.values()]  # Convert values to sets
            
        # Handle list input
        elif isinstance(data, list):
            # Validate non-empty list
            if len(data) == 0:
                raise ValueError("Input data cannot be empty")
            
            # Check if it's a list of (name, set) tuples
            if isinstance(data[0], tuple) and len(data[0]) == 2:
                # Extract names and sets from tuples
                self.sets_names = [item[0] for item in data]
                self.sets_list = [set(item[1]) for item in data]
            else:
                # Regular list of sets - auto-generate names
                self.sets_names = [f"Set{i+1}" for i in range(len(data))]
                self.sets_list = [set(s) for s in data]
        else:
            # Unsupported data format
            raise TypeError(f"Unsupported input type: {type(data)}")
    
    def _validate_data(self) -> None:
        """
        Validate the parsed data to ensure it meets requirements
        
        Performs the following checks:
        1. Number of names matches number of sets
        2. At least one set is provided
        3. All set names are unique
        
        Raises
        ------
        ValueError
            If any validation check fails
        """
        # Check 1: Names and sets count must match
        if len(self.sets_list) != len(self.sets_names):
            raise ValueError("Number of set names and sets do not match")
        
        # Check 2: Must have at least one set
        if len(self.sets_list) == 0:
            raise ValueError("At least one set must be provided")
        
        # Check 3: All set names must be unique
        if len(self.sets_names) != len(set(self.sets_names)):
            raise ValueError("Set names must be unique - duplicates found")
    
    def compute(self, min_size: int = 0, max_size: Optional[int] = None) -> pd.DataFrame:
        """
        Compute all overlaps between sets and return as DataFrame
        
        This is the core method that calculates all possible intersections between
        sets. It generates all 2^n - 1 non-empty combinations of sets, computes their
        intersections, and filters based on size criteria.
        
        The results are cached in self._results_df to avoid recomputation.
        
        Parameters
        ----------
        min_size : int, default 0
            Minimum overlap size threshold. Only returns combinations with 
            intersection size >= min_size. Set to 0 to include empty intersections.
        max_size : int, optional
            Maximum overlap size threshold. Only returns combinations with 
            intersection size <= max_size
            
        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - sets: tuple of set names
            - set_names: string representation (e.g., "Set1 & Set2 & Set3")
            - n_sets: number of sets in this combination
            - size: size of intersection
            - elements: sorted list of intersecting elements
            - exclusive_size: elements only in this combination
            - exclusive_elements: sorted list of exclusive elements
            
        Examples
        --------
        >>> calc = OverlapCalculator({'Set1': {1,2,3}, 'Set2': {2,3,4}})
        >>> 
        >>> # Get all combinations (including empty)
        >>> df = calc.compute(min_size=0)
        >>> print(f"Total combinations: {len(df)}")  # Should be 2^n - 1
        >>> 
        >>> # Get only non-empty overlaps
        >>> df = calc.compute(min_size=1)
        >>> print(df[['set_names', 'size', 'elements']])
        
        Notes
        -----
        - Results are cached - calling compute() again returns cached results
          unless different parameters are provided
        - With min_size=0, returns all 2^n - 1 possible combinations
        - With min_size=1 (default), excludes empty intersections
        - Empty intersections have size=0 and elements=[]
        """
        results = []  # Collect results for DataFrame construction
        n = len(self.sets_list)  # Total number of sets
        
        # Generate all possible combinations: choose r sets from n total
        # r ranges from 1 (single sets) to n (all sets together)
        for r in range(1, n + 1):
            # combinations(range(n), r) generates all r-length combinations of indices
            for idxs in combinations(range(n), r):
                # Compute intersection of all sets in this combination
                # Using set.intersection with unpacked generator
                intersect_set = set.intersection(*[self.sets_list[i] for i in idxs])
                
                # Apply size filtering
                # Note: When min_size=0, this includes empty intersections
                if len(intersect_set) < min_size:
                    continue  # Skip if intersection too small
                if max_size is not None and len(intersect_set) > max_size:
                    continue  # Skip if intersection too large
                
                # Get names of sets in this combination
                set_names = tuple(self.sets_names[i] for i in idxs)
                
                # Compute exclusive elements (elements only in this combination)
                # Elements in intersection but NOT in any other sets
                if intersect_set and r < n:  # Only compute if not empty and not all sets
                    # Get all other sets not in this combination
                    other_sets = [self.sets_list[i] for i in range(n) if i not in idxs]
                    # Union of all other sets
                    union_others = set.union(*other_sets) if other_sets else set()
                    # Exclusive = intersection - others
                    exclusive_set = intersect_set - union_others
                else:
                    # If empty or all sets are included, handle accordingly
                    exclusive_set = set() if not intersect_set else intersect_set
                
                # Store results for this combination
                results.append({
                    "sets": set_names,  # Tuple of set names
                    "set_names": " & ".join(set_names),  # Human-readable name
                    "n_sets": r,  # Number of sets in combination
                    "size": len(intersect_set),  # Intersection size
                    "elements": sorted(intersect_set),  # Sorted list of elements
                    "exclusive_size": len(exclusive_set),  # Exclusive elements count
                    "exclusive_elements": sorted(exclusive_set)  # Sorted exclusive elements
                })
        
        # Convert results list to DataFrame
        self._results_df = pd.DataFrame(results)
        
        # Sort results: first by number of sets (descending), then by size (descending)
        # This puts most interesting overlaps (many sets, large intersections) first
        if not self._results_df.empty:
            self._results_df = self._results_df.sort_values(
                ['n_sets', 'size'], 
                ascending=[False, False]  # Descending order for both
            ).reset_index(drop=True)  # Reset index after sorting
        
        return self._results_df
    
    def compute_exclusive(self) -> pd.DataFrame:
        """
        Compute exclusive elements for each individual set
        
        Exclusive elements are those that belong to only ONE set and not to any
        other sets in the collection. This method calculates these for each set.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - set: set name
            - total_size: total elements in the set
            - exclusive_size: elements only in this set
            - exclusive_elements: sorted list of exclusive elements
            - overlap_size: elements shared with other sets
            
        Notes
        -----
        Results are cached in self._exclusive_results
        overlap_size = total_size - exclusive_size
        """
        results = []  # Collect results for DataFrame construction
        n = len(self.sets_list)  # Total number of sets
        
        # Process each set individually
        for i in range(n):
            # Get current set
            current_set = self.sets_list[i]
            
            # Get all other sets (excluding current)
            other_sets = [self.sets_list[j] for j in range(n) if j != i]
            
            # Union of all other sets' elements
            union_others = set.union(*other_sets) if other_sets else set()
            
            # Exclusive elements = current_set - all_others
            exclusive = current_set - union_others
            
            # Store results for this set
            results.append({
                "set": self.sets_names[i],  # Set name
                "total_size": len(current_set),  # Total elements
                "exclusive_size": len(exclusive),  # Exclusive count
                "exclusive_elements": sorted(exclusive),  # Sorted list
                "overlap_size": len(current_set) - len(exclusive)  # Shared elements
            })
        
        # Convert to DataFrame and cache
        self._exclusive_results = pd.DataFrame(results)
        return self._exclusive_results
    
    def get_pairwise_overlap(self) -> Dict[str, pd.DataFrame]:
        """
        Generate pairwise overlap and Jaccard similarity matrices
        
        This method creates two symmetric matrices showing relationships between
        all pairs of sets:
        1. Overlap matrix: counts of intersecting elements
        2. Jaccard matrix: Jaccard similarity coefficients (0-1)
        
        Jaccard similarity = |A ∩ B| / |A ∪ B|
        
        Returns
        -------
        dict
            Dictionary containing:
            - 'overlap_matrix': DataFrame with intersection counts
            - 'jaccard_matrix': DataFrame with Jaccard coefficients
            
        Examples
        --------
        >>> matrices = calc.get_pairwise_overlap()
        >>> print(matrices['overlap_matrix'])
        >>> print(matrices['jaccard_matrix'])
        """
        n = len(self.sets_names)  # Number of sets
        
        # Initialize matrices with zeros
        # Index and columns are set names for easy lookup
        overlap_matrix = pd.DataFrame(0, 
                                    index=self.sets_names, 
                                    columns=self.sets_names)
        
        jaccard_matrix = pd.DataFrame(0.0, 
                                     index=self.sets_names, 
                                     columns=self.sets_names)
        
        # Calculate pairwise metrics
        # Only need to compute upper triangle due to symmetry
        for i in range(n):
            for j in range(i, n):
                set_i = self.sets_list[i]
                set_j = self.sets_list[j]
                
                # Intersection size
                intersection = len(set_i & set_j)
                
                # Union size
                union = len(set_i | set_j)
                
                # Fill both triangles (matrix is symmetric)
                overlap_matrix.iloc[i, j] = intersection
                overlap_matrix.iloc[j, i] = intersection
                
                # Jaccard similarity (handle division by zero)
                jaccard = intersection / union if union > 0 else 0.0
                jaccard_matrix.iloc[i, j] = jaccard
                jaccard_matrix.iloc[j, i] = jaccard
        
        return {
            'overlap_matrix': overlap_matrix,  # Raw intersection counts
            'jaccard_matrix': jaccard_matrix   # Similarity coefficients (0.0 to 1.0)
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Generate comprehensive summary statistics of all set overlaps
        
        This method provides a high-level overview including:
        - Basic counts and sizes
        - Elements common to all sets
        - Maximum overlap information
        - Complete overlap data in dictionary format
        - Statistics about empty vs non-empty combinations
        
        Returns
        -------
        dict
            Dictionary containing:
            - 'n_sets': number of input sets
            - 'set_names': list of all set names
            - 'set_sizes': dictionary mapping names to sizes
            - 'total_unique_elements': size of union of all sets
            - 'all_common_size': elements in all sets (if any)
            - 'max_overlap_size': size of largest overlap
            - 'max_overlap_sets': which sets have max overlap
            - 'all_overlaps': list of overlap dictionaries from last compute()
            - 'n_combinations': number of combinations from last compute()
            - 'empty_combinations': number of empty intersections
            - 'non_empty_combinations': number of non-empty intersections
            
        Notes
        -----
        Automatically triggers compute() if not already done
        Returns statistics based on whatever the last compute() produced
        Call compute(min_size=0) explicitly before get_summary() if you need all 2^n - 1 combinations
        """
        # Ensure computation is done
        # Note: This simply returns cached results. Call compute(min_size=0) explicitly
        # before get_summary() if you need all 2^n - 1 combinations including empty ones.
        if self._results_df is None:
            self.compute()
        
        # Build summary dictionary
        summary = {
            'n_sets': len(self.sets_list),  # Total number of sets
            'set_names': self.sets_names,   # List of all set names
            'set_sizes': {  # Dictionary of set sizes
                name: len(s) 
                for name, s in zip(self.sets_names, self.sets_list)
            },
            'total_unique_elements': len(set.union(*self.sets_list)),  # Union size
        }
        
        # Add overlap-specific statistics (now always available with min_size=0)
        # Elements common to ALL sets (intersection of all)
        all_common = self._results_df[
            self._results_df['n_sets'] == len(self.sets_list)
        ]
        summary['all_common_size'] = all_common['size'].iloc[0] if not all_common.empty else 0
        
        # Find maximum overlap (largest intersection)
        summary['max_overlap_size'] = self._results_df['size'].max()
        summary['max_overlap_sets'] = self._results_df.loc[
            self._results_df['size'].idxmax(), 'sets'
        ]
        
        # Include ALL 2^n - 1 combinations as list of dictionaries
        summary['all_overlaps'] = self._results_df.to_dict('records')
        summary['n_combinations'] = len(self._results_df)  # Total number of combinations
        summary['empty_combinations'] = len(self._results_df[self._results_df['size'] == 0])
        summary['non_empty_combinations'] = len(self._results_df[self._results_df['size'] > 0])
        
        return summary
    
    def get_dataframe(self) -> pd.DataFrame:
        """
        Get the computed overlap results as a DataFrame
        
        Returns a DataFrame with the results from the most recent compute() call.
        This is a simple getter that does not recompute or modify data.
        
        Returns
        -------
        pd.DataFrame
            DataFrame containing overlap computation results
            
        Notes
        -----
        Automatically triggers compute() if not already done
        Returns whatever the last compute() produced (including any filters)
        """
        # Ensure computation is done
        # Note: This simply returns cached results. Call compute(min_size=0) explicitly
        # before get_dataframe() if you need all 2^n - 1 combinations including empty ones.
        if self._results_df is None:
            self.compute()
        return self._results_df
    
    def get_plot_data(self) -> pd.DataFrame:
        """
        Get DataFrame formatted for Venn diagram plotting
        
        Returns a DataFrame with selected columns suitable for plotting Venn diagrams.
        For single-set regions (n_sets == 1), shows exclusive elements instead of 
        all elements, which is more appropriate for Venn diagram region labeling.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with columns: set_names, n_sets, size, elements
            For n_sets == 1: size = exclusive_size, elements = exclusive_elements
            For n_sets > 1: size and elements are the intersection values
            
        Notes
        -----
        Automatically triggers compute(min_size=0) if not already done
        This ensures all 2^n - 1 combinations are available for plotting
        """
        # Ensure we have computed data with min_size=0 to include all combinations
        if self._results_df is None:
            self.compute(min_size=0)
        
        # Create a copy to avoid modifying the original cached DataFrame
        plot_df = self._results_df[['set_names', 'n_sets', 'size', 'elements', 'exclusive_size', 'exclusive_elements']].copy()
        
        # For single-set regions (n_sets == 1), replace with exclusive elements
        # This makes Venn diagrams show region-specific elements correctly
        single_set_mask = plot_df['n_sets'] == 1
        plot_df.loc[single_set_mask, 'size'] = plot_df.loc[single_set_mask, 'exclusive_size']
        plot_df.loc[single_set_mask, 'elements'] = plot_df.loc[single_set_mask, 'exclusive_elements']
        
        # Drop the exclusive columns as they're no longer needed
        plot_df = plot_df.drop(columns=['exclusive_size', 'exclusive_elements'])
        
        return plot_df
    
    def query_elements(self, query_sets: Optional[List[str]] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Query overlap information for specific set combinations
        
        This flexible method allows querying specific set combinations or
        retrieving all possible combinations when no argument is provided.
        
        Parameters
        ----------
        query_sets : list of str, optional
            List of set names to query. If None (default), returns ALL 
            possible overlap combinations in the dataset.
            
        Returns
        -------
        dict or list of dict
            - If query_sets provided: Single dictionary with overlap info
            - If query_sets is None: List of dictionaries for all combinations
            
        Dictionary keys:
        - 'sets': tuple of set names
        - 'set_names': string representation
        - 'n_sets': number of sets in combination
        - 'size': intersection size
        - 'elements': sorted list of intersecting elements
        - 'exclusive_size': exclusive elements count
        - 'exclusive_elements': sorted exclusive elements
        
        Examples
        --------
        >>> # Query specific combination
        >>> result = calc.query_elements(['SetA', 'SetB'])
        >>> print(result['elements'])
        
        >>> # Get all combinations
        >>> all_results = calc.query_elements()  # No arguments
        >>> for combo in all_results:
        >>>     print(f"{combo['set_names']}: {combo['elements']}")
        """
        # If no specific sets provided, return all possible combinations
        if query_sets is None:
            results = []
            n = len(self.sets_list)
            
            # Generate all possible combinations
            for r in range(1, n + 1):
                for idxs in combinations(range(n), r):
                    intersect_set = set.intersection(*[self.sets_list[i] for i in idxs])
                    
                    
                    
                    set_names = tuple(self.sets_names[i] for i in idxs)
                    
                    # Compute exclusive elements (only for non-empty intersections)
                    if intersect_set and r < n:
                        other_sets = [self.sets_list[i] for i in range(n) if i not in idxs]
                        union_others = set.union(*other_sets) if other_sets else set()
                        exclusive_set = intersect_set - union_others
                    else:
                        exclusive_set = set()
                    
                    results.append({
                        'sets': set_names,
                        'set_names': " & ".join(set_names),
                        'n_sets': r,
                        'size': len(intersect_set),
                        'elements': sorted(intersect_set),
                        'exclusive_size': len(exclusive_set),
                        'exclusive_elements': sorted(exclusive_set)
                    })
            
            return results
        
        # Validate set names
        for name in query_sets:
            if name not in self.sets_names:
                raise ValueError(f"Unknown set name: {name}")
        
        # Get indices
        idxs = [self.sets_names.index(name) for name in query_sets]
        
        # Compute intersection
        intersect_set = set.intersection(*[self.sets_list[i] for i in idxs])
        
        # Compute exclusive elements
        n = len(self.sets_list)
        r = len(idxs)
        if r < n:
            other_sets = [self.sets_list[i] for i in range(n) if i not in idxs]
            union_others = set.union(*other_sets) if other_sets else set()
            exclusive_set = intersect_set - union_others
        else:
            exclusive_set = intersect_set
        
        # Return dictionary with all overlap information
        return {
            'sets': tuple(query_sets),
            'set_names': " & ".join(query_sets),
            'n_sets': r,
            'size': len(intersect_set),
            'elements': sorted(intersect_set),
            'exclusive_size': len(exclusive_set),
            'exclusive_elements': sorted(exclusive_set)
        }
    
    def __repr__(self) -> str:
        """
        String representation of the calculator instance
        
        Returns
        -------
        str
            Formatted string showing number of sets and their names
            
        Examples
        --------
        >>> calc = OverlapCalculator({'Set1': {1,2}, 'Set2': {2,3}})
        >>> print(calc)
        OverlapCalculator(n_sets=2, sets=['Set1', 'Set2'])
        """
        return f"OverlapCalculator(n_sets={len(self.sets_list)}, sets={self.sets_names})"
