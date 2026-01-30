# calculator/overlap_calculator.py
import pandas as pd
from typing import List, Set, Dict, Tuple, Any
from itertools import combinations

class OverlapCalculator:
    def __init__(self, data: Any):
        if isinstance(data, dict):
            self.sets_names = list(data.keys())
            self.sets_list = list(data.values())
        else:
            self.sets_names = [f"Set{i+1}" for i in range(len(data))]
            self.sets_list = data
        
        self._results_df = None

    def compute(self) -> pd.DataFrame:
        results = []
        n = len(self.sets_list)
        for r in range(1, n + 1):
            for idxs in combinations(range(n), r):
                intersect_set = set.intersection(*[self.sets_list[i] for i in idxs])
                if intersect_set: 
                    set_names = tuple(self.sets_names[i] for i in idxs)
                    results.append({
                        "sets": set_names,
                        "size": len(intersect_set),
                        "elements": sorted(intersect_set)
                    })
        self._results_df = pd.DataFrame(results)
        return self._results_df

    def get_dataframe(self) -> pd.DataFrame:
        if self._results_df is None:
            self.compute()
        return self._results_df
