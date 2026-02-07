"""3D Voxel Costmap for Foresight - visually impaired navigation."""

from __future__ import annotations

import numpy as np
from typing import Tuple, List, Optional
from foresight.config import (
    CostmapConfig,
    DEFAULT_CONFIG,
    GROUND_LAYER,
    TORSO_LAYER,
    HEAD_LAYER,
)


class VoxelCostmap:
    """
    3D occupancy grid for obstacle detection at multiple heights.
    Voxel states: 0=free, 50=unknown, 75=inflated, 100=occupied
    """

    def __init__(
        self,
        origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        config: Optional[CostmapConfig] = None,
    ):
        self.config = config or DEFAULT_CONFIG
        self.origin = np.array(origin)
        nx, ny, nz = self.config.num_voxels
        self.grid = np.full((nx, ny, nz), self.config.UNKNOWN, dtype=np.int8)
        self.res = self.config.voxel_resolution

    def _world_to_voxel(self, x: float, y: float, z: float) -> Tuple[int, int, int]:
        """Convert world coords to voxel indices."""
        nx, ny, nz = self.config.num_voxels
        ix = int((x - self.origin[0]) / self.res)
        iy = int((y - self.origin[1]) / self.res)
        iz = int((z - self.origin[2]) / self.res)
        return (
            np.clip(ix, 0, nx - 1),
            np.clip(iy, 0, ny - 1),
            np.clip(iz, 0, nz - 1),
        )

    def _voxel_to_world(self, ix: int, iy: int, iz: int) -> Tuple[float, float, float]:
        """Convert voxel indices to world coords (center of voxel)."""
        return (
            self.origin[0] + (ix + 0.5) * self.res,
            self.origin[1] + (iy + 0.5) * self.res,
            self.origin[2] + (iz + 0.5) * self.res,
        )

    def _in_bounds(self, ix: int, iy: int, iz: int) -> bool:
        nx, ny, nz = self.config.num_voxels
        return 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz

    def add_point_cloud(
        self,
        points: np.ndarray,
        sensor_origin: Tuple[float, float, float],
        mark_free: bool = True,
    ) -> None:
        """
        Update costmap from 3D point cloud (e.g. from SLAM).
        points: (N, 3) array of [x,y,z] in world coords
        mark_free: if True, ray-trace from sensor to each point to mark free space
        """
        sx, sy, sz = self._world_to_voxel(sensor_origin[0], sensor_origin[1], sensor_origin[2])

        for p in points:
            if len(p) < 3:
                continue
            px, py, pz = self._world_to_voxel(float(p[0]), float(p[1]), float(p[2]))
            if mark_free:
                self._ray_trace(sx, sy, sz, px, py, pz)
            self.grid[px, py, pz] = self.config.OCCUPIED

        self._inflate_obstacles()

    def _ray_trace(self, x0: int, y0: int, z0: int, x1: int, y1: int, z1: int) -> None:
        """Mark voxels along ray as free (stop at first occupied)."""
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        dz = abs(z1 - z0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        sz = 1 if z0 < z1 else -1

        if dx >= dy and dx >= dz:
            err_y, err_z = 2 * dy - dx, 2 * dz - dx
            x, y, z = x0, y0, z0
            for _ in range(dx + 1):
                if self._in_bounds(x, y, z):
                    if self.grid[x, y, z] != self.config.OCCUPIED:
                        self.grid[x, y, z] = self.config.FREE
                    else:
                        return
                if err_y > 0:
                    y += sy
                    err_y -= 2 * dx
                if err_z > 0:
                    z += sz
                    err_z -= 2 * dx
                err_y += 2 * dy
                err_z += 2 * dz
                x += sx
        elif dy >= dx and dy >= dz:
            err_x, err_z = 2 * dx - dy, 2 * dz - dy
            x, y, z = x0, y0, z0
            for _ in range(dy + 1):
                if self._in_bounds(x, y, z):
                    if self.grid[x, y, z] != self.config.OCCUPIED:
                        self.grid[x, y, z] = self.config.FREE
                    else:
                        return
                if err_x > 0:
                    x += sx
                    err_x -= 2 * dy
                if err_z > 0:
                    z += sz
                    err_z -= 2 * dy
                err_x += 2 * dx
                err_z += 2 * dz
                y += sy
        else:
            err_x, err_y = 2 * dx - dz, 2 * dy - dz
            x, y, z = x0, y0, z0
            for _ in range(dz + 1):
                if self._in_bounds(x, y, z):
                    if self.grid[x, y, z] != self.config.OCCUPIED:
                        self.grid[x, y, z] = self.config.FREE
                    else:
                        return
                if err_x > 0:
                    x += sx
                    err_x -= 2 * dz
                if err_y > 0:
                    y += sy
                    err_y -= 2 * dz
                err_x += 2 * dx
                err_y += 2 * dy
                z += sz

    def add_obstacle(self, x: float, y: float, z: float) -> None:
        """Mark a single point as occupied."""
        ix, iy, iz = self._world_to_voxel(x, y, z)
        if self._in_bounds(ix, iy, iz):
            self.grid[ix, iy, iz] = self.config.OCCUPIED
            self._inflate_obstacles()

    def add_obstacle_box(
        self,
        x_min: float, y_min: float, z_min: float,
        x_max: float, y_max: float, z_max: float,
    ) -> None:
        """Mark a 3D box region as occupied."""
        ix0, iy0, iz0 = self._world_to_voxel(x_min, y_min, z_min)
        ix1, iy1, iz1 = self._world_to_voxel(x_max, y_max, z_max)
        nx, ny, nz = self.config.num_voxels
        for ix in range(max(0, ix0), min(nx, ix1 + 1)):
            for iy in range(max(0, iy0), min(ny, iy1 + 1)):
                for iz in range(max(0, iz0), min(nz, iz1 + 1)):
                    self.grid[ix, iy, iz] = self.config.OCCUPIED
        self._inflate_obstacles()

    def _inflate_obstacles(self) -> None:
        """Add safety buffer around occupied voxels."""
        r = max(1, int(self.config.inflation_radius / self.res))
        occ = self.grid == self.config.OCCUPIED
        inflated = np.copy(self.grid)

        nx, ny, nz = self.config.num_voxels
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    if not occ[ix, iy, iz]:
                        continue
                    for di in range(-r, r + 1):
                        for dj in range(-r, r + 1):
                            for dk in range(-r, r + 1):
                                if di * di + dj * dj + dk * dk > r * r:
                                    continue
                                ni, nj, nk = ix + di, iy + dj, iz + dk
                                if self._in_bounds(ni, nj, nk):
                                    if inflated[ni, nj, nk] == self.config.FREE:
                                        inflated[ni, nj, nk] = self.config.INFLATED
                                    elif inflated[ni, nj, nk] == self.config.UNKNOWN:
                                        inflated[ni, nj, nk] = self.config.INFLATED

        self.grid = inflated

    def is_path_safe(
        self,
        start: Tuple[float, float, float],
        end: Tuple[float, float, float],
        body_height_range: Tuple[float, float] = TORSO_LAYER,
    ) -> bool:
        """
        Check if a path is safe for navigation within the given height range.
        Returns False if any voxel along the path in that height band is occupied or inflated.
        """
        z_min, z_max = body_height_range
        ix0, iy0, iz0 = self._world_to_voxel(start[0], start[1], start[2])
        ix1, iy1, iz1 = self._world_to_voxel(end[0], end[1], end[2])
        iz_min = max(0, int((z_min - self.origin[2]) / self.res))
        iz_max = min(self.config.num_voxels[2] - 1, int((z_max - self.origin[2]) / self.res))

        # Sample along 2D path (x,y) at multiple heights
        steps = max(
            abs(ix1 - ix0),
            abs(iy1 - iy0),
            1,
        ) * 2
        for t in np.linspace(0, 1, steps + 1):
            ix = int(ix0 + t * (ix1 - ix0))
            iy = int(iy0 + t * (iy1 - iy0))
            for iz in range(iz_min, iz_max + 1):
                if not self._in_bounds(ix, iy, iz):
                    continue
                v = self.grid[ix, iy, iz]
                if v in (self.config.OCCUPIED, self.config.INFLATED):
                    return False
        return True

    def get_layer_slice(
        self,
        layer: Tuple[float, float],
        axis: str = "z",
    ) -> np.ndarray:
        """Get 2D slice of costmap for a height layer. axis='z' returns xy plane."""
        z_min, z_max = layer
        iz0 = max(0, int((z_min - self.origin[2]) / self.res))
        iz1 = min(self.config.num_voxels[2], int((z_max - self.origin[2]) / self.res) + 1)
        if axis == "z":
            return np.max(self.grid[:, :, iz0:iz1], axis=2)
        elif axis == "y":
            return np.max(self.grid[:, iz0:iz1, :], axis=1)
        else:
            return np.max(self.grid[iz0:iz1, :, :], axis=0)

    def get_occupied_voxels(
        self,
        include_inflated: bool = True,
        layer: Optional[Tuple[float, float]] = None,
    ) -> np.ndarray:
        """Return (N, 3) array of [x,y,z] for occupied/inflated voxels (for visualization)."""
        mask = (self.grid == self.config.OCCUPIED) | (
            (self.grid == self.config.INFLATED) if include_inflated else np.zeros_like(self.grid, dtype=bool)
        )
        if layer:
            z_min, z_max = layer
            iz0 = max(0, int((z_min - self.origin[2]) / self.res))
            iz1 = min(self.config.num_voxels[2], int((z_max - self.origin[2]) / self.res) + 1)
            mask[:, :, :iz0] = False
            mask[:, :, iz1:] = False
        ixs, iys, izs = np.where(mask)
        coords = np.column_stack([
            self.origin[0] + (ixs + 0.5) * self.res,
            self.origin[1] + (iys + 0.5) * self.res,
            self.origin[2] + (izs + 0.5) * self.res,
        ])
        return coords
