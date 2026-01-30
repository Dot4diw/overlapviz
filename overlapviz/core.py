# core.py
import pandas as pd
from typing import List, Set, Dict, Any, Union

class OverlapViz:
    def __init__(self, data: Union[List[Set], Dict[str, Set]]):
        if isinstance(data, dict):
            self.sets_names = list(data.keys())
            self.sets_list = list(data.values())
        else:
            self.sets_names = [f"Set{i+1}" for i in range(len(data))]
            self.sets_list = data
        
        self._overlap_results = None
        self._last_plot_type = None

    def _compute_overlaps(self):
        from itertools import combinations
        all_elements = set().union(*self.sets_list)
        results = []
        for r in range(1, len(self.sets_list) + 1):
            for idxs in combinations(range(len(self.sets_list)), r):
                intersect_set = set.intersection(*[self.sets_list[i] for i in idxs])
                if intersect_set:
                    set_names = tuple(self.sets_names[i] for i in idxs)
                    results.append({
                        "sets": set_names,
                        "size": len(intersect_set),
                        "elements": sorted(intersect_set)
                    })
        self._overlap_results = pd.DataFrame(results)

    def get(self) -> pd.DataFrame:
        if self._overlap_results is None:
            self._compute_overlaps()
        return self._overlap_results

    def plot_venn(self, **kwargs):
        from .venn.plotter import VennPlotter
        self._last_plot_type = "venn"
        return VennPlotter(self).plot(**kwargs)

    def plot_euler(self, **kwargs):
        from .euler.plotter import EulerPlotter
        self._last_plot_type = "euler"
        return EulerPlotter(self).plot(**kwargs)

    def plot_upset(self, **kwargs):
        from .upset.plotter import UpsetPlotter
        self._last_plot_type = "upset"
        return UpsetPlotter(self).plot(**kwargs)
