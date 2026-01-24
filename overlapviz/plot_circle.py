import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.optimize import minimize
import sys

def circle_radius(area):
    """Calculate radius from area: r = sqrt(A/pi)"""
    return np.sqrt(area / np.pi)


def intersection_area(d, r1, r2):
    """
    Calculate intersection area of two circles
    d: distance between centers, r1, r2: radii
    """
    if d >= r1 + r2:
        return 0
    elif d <= abs(r1 - r2):
        return min(np.pi * r1**2, np.pi * r2**2)
    
    alpha = np.arccos((d**2 + r1**2 - r2**2) / (2 * d * r1))
    beta = np.arccos((d**2 + r2**2 - r1**2) / (2 * d * r2))
    
    return (r1**2 * alpha + r2**2 * beta - 
            0.5 * r1**2 * np.sin(2*alpha) - 
            0.5 * r2**2 * np.sin(2*beta))


# ==================== Base Venn Diagram Class ====================

class VennDiagram:
    """Base class for Venn diagrams"""
    
    def __init__(self, subsets, labels, colors=None):
        """
        Initialize Venn diagram
        
        Args:
            subsets: List of region sizes
                     For Venn2: [A, B, AB]
                     For Venn3: [A, B, C, AB, BC, AC, ABC]
            labels: Tuple of set labels
            colors: List of colors for each set
        """
        self.subsets = np.array(subsets, dtype=float)
        self.labels = labels
        self.colors = colors if colors else self._default_colors(len(labels))
        self.centers = None
        self.radii = None
    
    @staticmethod
    def _default_colors(n):
        """Generate default colors"""
        default_palette = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#F7DC6F', '#BB8FCE']
        return default_palette[:n]
    
    def calculate_radii(self):
        """Calculate radii from subset areas - to be implemented by subclasses"""
        raise NotImplementedError
    
    def estimate_areas(self, centers):
        """Estimate region areas for given centers - to be implemented by subclasses"""
        raise NotImplementedError
    
    def solve_positions(self):
        """Solve for optimal circle positions using numerical optimization"""
        self.radii = self.calculate_radii()
        n_circles = len(self.radii)
        
        # Initialize positions
        x0 = self._initialize_positions()
        
        # Desired areas (avoid zeros)
        desired = np.maximum(self.subsets, 0.001)
        
        # Cost function
        def cost(x):
            centers = x.reshape((n_circles, 2))
            try:
                current = self.estimate_areas(centers)
                current = np.maximum(current, 0.001)
                log_diff = np.log(current) - np.log(desired)
                return np.sum(log_diff**2)
            except:
                return 1e10
        
        # Optimize
        result = minimize(cost, x0, method='Nelder-Mead', 
                         options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
        
        self.centers = result.x.reshape((n_circles, 2))
        return self.centers, self.radii
    
    def _initialize_positions(self):
        """Initialize circle positions - to be implemented by subclasses"""
        raise NotImplementedError
    
    def draw(self, ax, show_labels=True):
        """Draw the Venn diagram on given axes"""
        if self.centers is None or self.radii is None:
            self.solve_positions()
        
        # Draw circles
        for i, (center, radius, color, label) in enumerate(
            zip(self.centers, self.radii, self.colors, self.labels)
        ):
            circle = Circle(center, radius, alpha=0.5, color=color)
            ax.add_patch(circle)
            
            # Add label at center
            if show_labels:
                ax.annotate(label, xy=center, fontsize=14,
                           ha='center', va='center',
                           fontweight='bold', color='white')
        
        # Set bounds
        self._set_bounds(ax)
        ax.set_aspect('equal')
        ax.axis('off')
        
        return self.centers, self.radii
    
    def _set_bounds(self, ax):
        """Set axis bounds - to be implemented by subclasses"""
        raise NotImplementedError


# ==================== Venn2 Implementation ====================

class Venn2(VennDiagram):
    """2-set Venn diagram implementation"""
    
    def calculate_radii(self):
        """Calculate radii for 2-set Venn diagram"""
        A, B, AB = self.subsets
        
        area_A = A + AB
        area_B = B + AB
        
        return [circle_radius(area_A), circle_radius(area_B)]
    
    def estimate_areas(self, centers):
        """Estimate 3 region areas for 2-set Venn diagram"""
        centers = np.array(centers)
        rA, rB = self.radii
        
        # Distance between centers
        d_AB = np.linalg.norm(centers[0] - centers[1])
        
        # Intersection area
        area_AB = intersection_area(d_AB, rA, rB)
        
        # Individual regions
        area_A = np.pi * rA**2 - area_AB
        area_B = np.pi * rB**2 - area_AB
        
        return [area_A, area_B, area_AB]
    
    def _initialize_positions(self):
        """Initialize 2 circles side by side"""
        rA, rB = self.radii
        d_init = (rA + rB) * 0.7  # Start with some overlap
        return np.array([
            [-d_init * 0.5, 0],
            [d_init * 0.5, 0]
        ]).flatten()
    
    def _set_bounds(self, ax):
        """Set bounds for 2-set Venn diagram"""
        all_pts = np.array(self.centers)
        max_radius = max(self.radii)
        
        mins = all_pts.min(axis=0) - max_radius
        maxs = all_pts.max(axis=0) + max_radius
        
        margin = max_radius * 0.3
        ax.set_xlim(mins[0] - margin, maxs[0] + margin)
        ax.set_ylim(mins[1] - margin, maxs[1] + margin)


# ==================== Venn3 Implementation ====================

class Venn3(VennDiagram):
    """3-set Venn diagram implementation"""
    
    def calculate_radii(self):
        """Calculate radii for 3-set Venn diagram"""
        A, B, C, AB, BC, AC, ABC = self.subsets
        
        area_A = A + AB + AC + ABC
        area_B = B + AB + BC + ABC
        area_C = C + AC + BC + ABC
        
        return [circle_radius(area_A), circle_radius(area_B), circle_radius(area_C)]
    
    def estimate_areas(self, centers):
        """Estimate 7 region areas for 3-set Venn diagram"""
        centers = np.array(centers)
        rA, rB, rC = self.radii
        
        # Pairwise distances and intersections
        d_AB = np.linalg.norm(centers[0] - centers[1])
        d_BC = np.linalg.norm(centers[1] - centers[2])
        d_AC = np.linalg.norm(centers[0] - centers[2])
        
        area_AB = intersection_area(d_AB, rA, rB)
        area_BC = intersection_area(d_BC, rB, rC)
        area_AC = intersection_area(d_AC, rA, rC)
        
        # Estimate triple intersection (simplified)
        area_ABC = min(area_AB, area_BC, area_AC) * 0.3
        
        # Individual regions
        area_A = np.pi * rA**2 - area_AB - area_AC + area_ABC
        area_B = np.pi * rB**2 - area_AB - area_BC + area_ABC
        area_C = np.pi * rC**2 - area_AC - area_BC + area_ABC
        
        return [area_A, area_B, area_C, 
                area_AB - area_ABC, area_BC - area_ABC, area_AC - area_ABC,
                area_ABC]
    
    def _initialize_positions(self):
        """Initialize 3 circles in triangular arrangement"""
        rA, rB, rC = self.radii
        d_init = (rA + rB + rC) / 3
        return np.array([
            [-d_init * 0.5, -d_init * 0.3],
            [d_init * 0.5, -d_init * 0.3],
            [0, d_init * 0.6]
        ]).flatten()
    
    def _set_bounds(self, ax):
        """Set bounds for 3-set Venn diagram"""
        all_pts = np.array(self.centers)
        max_radius = max(self.radii)
        
        mins = all_pts.min(axis=0) - max_radius
        maxs = all_pts.max(axis=0) + max_radius
        
        margin = max_radius * 0.2
        ax.set_xlim(mins[0] - margin, maxs[0] + margin)
        ax.set_ylim(mins[1] - margin, maxs[1] + margin)


# ==================== Factory Function ====================

def create_venn(subsets, labels, colors=None):
    """
    Factory function to create appropriate Venn diagram
    
    Args:
        subsets: List of region sizes
                 For Venn2: [A, B, AB] (3 elements)
                 For Venn3: [A, B, C, AB, BC, AC, ABC] (7 elements)
        labels: Tuple of set labels (2 or 3 elements)
        colors: Optional list of colors
    
    Returns:
        VennDiagram instance
    """
    n_regions = len(subsets)
    n_labels = len(labels)
    
    if n_regions == 3 and n_labels == 2:
        return Venn2(subsets, labels, colors)
    elif n_regions == 7 and n_labels == 3:
        return Venn3(subsets, labels, colors)
    else:
        raise ValueError(
            f"Unsupported combination: {n_regions} regions with {n_labels} labels. "
            "Expected (3 regions, 2 labels) for Venn2 or (7 regions, 3 labels) for Venn3"
        )


# ==================== Demo ====================

def demo_venn2():
    """Demonstrate 2-set Venn diagrams"""
    print("\n" + "="*50)
    print("2-SET VENN DIAGRAM DEMO")
    print("="*50)
    
    examples = [
        {
            'name': 'Basic 2-Set',
            'subsets': [5, 7, 3],  # [A, B, AB]
            'labels': ('A', 'B'),
            'colors': ['#FF6B6B', '#4ECDC4']
        },
        {
            'name': 'High Overlap',
            'subsets': [2, 3, 8],  # More in intersection
            'labels': ('X', 'Y'),
            'colors': ['#9B59B6', '#3498DB']
        },
        {
            'name': 'No Overlap',
            'subsets': [6, 4, 0],  # Disjoint sets
            'labels': ('M', 'N'),
            'colors': ['#E74C3C', '#F39C12']
        }
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (ax, ex) in enumerate(zip(axes, examples)):
        print(f"Drawing 2-set example {i+1}: {ex['name']}")
        print(f"  Input: A={ex['subsets'][0]}, B={ex['subsets'][1]}, AB={ex['subsets'][2]}")
        
        venn = create_venn(ex['subsets'], ex['labels'], ex['colors'])
        centers, radii = venn.draw(ax)
        
        print(f"  Radii: {radii}")
        print(f"  Centers: {centers}")
        
        ax.set_title(f"{ex['name']}\nA={ex['subsets'][0]}, B={ex['subsets'][1]}, AB={ex['subsets'][2]}", 
                    fontsize=10)
    
    plt.tight_layout()
    plt.savefig('venn2_demo.png', dpi=150, bbox_inches='tight')
    print("Saved: venn2_demo.png")


def demo_venn3():
    """Demonstrate 3-set Venn diagrams"""
    print("\n" + "="*50)
    print("3-SET VENN DIAGRAM DEMO")
    print("="*50)
    
    examples = [
        {
            'name': 'Equal Sets',
            'subsets': [3, 3, 3, 2, 2, 2, 1],
            'labels': ('A', 'B', 'C'),
            'colors': ['#FF6B6B', '#4ECDC4', '#45B7D1']
        },
        {
            'name': 'Large A',
            'subsets': [10, 2, 2, 4, 1, 1, 0.5],
            'labels': ('Big', 'Medium', 'Small'),
            'colors': ['#E74C3C', '#F39C12', '#2ECC71']
        },
        {
            'name': 'High Overlap',
            'subsets': [1, 1, 1, 3, 3, 3, 2],
            'labels': ('X', 'Y', 'Z'),
            'colors': ['#9B59B6', '#3498DB', '#1ABC9C']
        }
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (ax, ex) in enumerate(zip(axes, examples)):
        print(f"Drawing 3-set example {i+1}: {ex['name']}")
        print(f"  Input: {ex['subsets']}")
        
        venn = create_venn(ex['subsets'], ex['labels'], ex['colors'])
        centers, radii = venn.draw(ax)
        
        print(f"  Radii: {radii}")
        print(f"  Centers: {centers}")
        
        ax.set_title(f"{ex['name']}\n{ex['subsets']}", fontsize=10)
    
    plt.tight_layout()
    plt.savefig('venn3_demo.png', dpi=150, bbox_inches='tight')
    print("Saved: venn3_demo.png")


def main():
    """Run comprehensive demonstration"""
    print("Starting Universal Venn Diagram demonstration...")

    # Run demos
    demo_venn2()
    demo_venn3()
    
    print("\n" + "="*50)
    print("DEMONSTRATION COMPLETED!")
    print("="*50)
    print("Generated files:")
    print("  - venn2_demo.png (2-set examples)")
    print("  - venn3_demo.png (3-set examples)")
    print("\nUsage:")
    print("  venn = create_venn([5, 7, 3], ('A', 'B'))  # 2-set")
    print("  venn = create_venn([3,3,3,2,2,2,1], ('A','B','C'))  # 3-set")




if __name__ == "__main__":
    main()
