"""Foresight - 3D costmap and SLAM configuration."""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class CostmapConfig:
    """3D costmap specifications for visually impaired navigation."""

    # Voxel resolution: 10cm cubes
    voxel_resolution: float = 0.1

    # Map size: width (x) × depth (y) × height (z) in meters
    map_size: Tuple[float, float, float] = (20.0, 20.0, 3.0)

    # Height range: ground to ceiling
    height_min: float = 0.0
    height_max: float = 3.0

    # Inflation safety buffer around obstacles (meters)
    inflation_radius: float = 0.3

    # Voxel states
    FREE: int = 0
    OCCUPIED: int = 100
    UNKNOWN: int = 50
    INFLATED: int = 75

    @property
    def num_voxels(self) -> Tuple[int, int, int]:
        """Grid dimensions in voxels."""
        return (
            int(self.map_size[0] / self.voxel_resolution),
            int(self.map_size[1] / self.voxel_resolution),
            int(self.map_size[2] / self.voxel_resolution),
        )


# Height layer boundaries for different body parts (meters)
GROUND_LAYER = (0.0, 0.3)   # Ankle obstacles
TORSO_LAYER = (0.3, 1.8)    # Main navigation obstacles
HEAD_LAYER = (1.8, 2.5)     # Overhead obstacles

DEFAULT_CONFIG = CostmapConfig()
