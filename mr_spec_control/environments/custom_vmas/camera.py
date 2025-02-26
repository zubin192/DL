import typing
from typing import Callable, List, Union, Tuple
import torch
from PIL import Image
import numpy as np

from vmas.simulator.core import Box, Entity, World
from vmas.simulator.sensors import Sensor
from vmas.simulator.utils import Color

from vmas.simulator import rendering

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom

class TopDownCamera(Sensor):
    def __init__(
        self,
        world: World,
        frame_x_dim: float,
        frame_y_dim: float,
        resolution: Tuple[int, int] = (64, 64),
        center: Union[None, torch.Tensor] = None,
        render_color: Union[Color, Tuple[float, float, float]] = Color.YELLOW,
        alpha: float = 0.1,
        entity_filter: Callable[[Entity], bool] = lambda _: True,
    ):
        """
        Creates a top-down camera sensor that captures a rendered view of the environment.

        Parameters:
        - world: The VMAS world object.
        - x_dim, y_dim: Dimensions of the camera view.
        - resolution: Output image resolution (width, height).
        - center: The center of the camera view (default is the agent's position).
        - render_color: Color of rendered entities.
        - alpha: Transparency of rendered entities.
        - entity_filter: Function to filter entities to render.
        """
        super().__init__(world)
        self.frame_x_dim = frame_x_dim
        self.frame_y_dim = frame_y_dim
        self.resolution = resolution
        self.center = center  # Defaults to agent position if None
        self.render_color = render_color
        self.alpha = alpha
        self.entity_filter = entity_filter

        if not center:
            center = torch.zeros((world.batch_dim, 2), device=world.device)

        self.viewers = [rendering.Viewer(self.resolution[0], self.resolution[1], visible=False) for _ in center]
        self.views = []
        for env, viewer in enumerate(self.viewers):
            viewer.set_bounds(
                center[env][0] - self.frame_x_dim / 2, center[env][0] + self.frame_x_dim / 2,
                center[env][1] - self.frame_y_dim / 2, center[env][1] + self.frame_y_dim / 2
            )
            self.views.append(viewer.render(return_rgb_array=True))

        self._last_image = torch.from_numpy(np.stack(self.views)).to(self._world.device)
        # self._last_image = torch.asarray(self.views, device=world.device)

    def to(self, device: torch.device):
        self._last_image = self._last_image.to(device)

    def measure(self, new_center=None):
        """Generates a top-down view of the environment as an image."""

        if new_center is None:
            center = self.agent.state.pos
        else:
            center = new_center

        # Create a top-down camera view
        for env, viewer in enumerate(self.viewers):
            viewer.set_bounds(
                center[env][0] - self.frame_x_dim / 2, center[env][0] + self.frame_x_dim / 2,
                center[env][1] - self.frame_y_dim / 2, center[env][1] + self.frame_y_dim / 2
            )

            # Render filtered entities in the scene
            for entity in self._world._agents + self._world._landmarks:
                if not self.entity_filter(entity):  # Apply entity filter
                    continue

                pos = entity.state.pos
                if (
                    center[env][0] - self.frame_x_dim / 2 <= pos[env][0] <= center[env][0] + self.frame_x_dim / 2
                    and center[env][1] - self.frame_y_dim / 2 <= pos[env][1] <= center[env][1] + self.frame_y_dim / 2
                ):
                    # Render if in scene region
                    geom = entity.shape.get_geometry()
                    # geom = rendering.make_circle(entity.size if hasattr(entity, "size") else 0.05)
                    geom.set_color(*entity.color, alpha=1.0)
                    xform = rendering.Transform()
                    xform.set_translation(pos[env][0], pos[env][1])
                    geom.add_attr(xform)
                    viewer.add_onetime(geom)

            # self.views[i] = viewer.render(return_rgb_array=True)

            # Render the image and return it as an array
            # TODO probably a more efficient way to do this
            self._last_image[env] = torch.from_numpy(viewer.render(return_rgb_array=True).copy()).to(self._world.device)
        # self._last_image = torch.asarray(self.views, device=self._world.device)

        # print(f"!! Last image shape: {self._last_image.shape}")
        return self._last_image

    def save_image(self, filename: str):
        """Saves the last rendered images to a file."""
        if self._last_image is None:
            raise ValueError("No image has been captured yet. Call measure() first.")

        for i, im_arr in enumerate(self._last_image):
            image = Image.fromarray(np.uint8(im_arr))  # Convert NumPy array to an image
            image.save(filename)
            print(f"Top-down view saved to {filename}")

    def render(self, env_index: int = 0) -> "List[Geom]":
        """Returns a rendering of the image view region"""
        if self._last_image is None:
            return []

        region = rendering.make_circle(self.frame_x_dim/2)
        region = Box(self.frame_x_dim/2, self.frame_y_dim/2).get_geometry()
        region.set_color(*self.render_color.value, self.alpha)
        pos_region = self.agent.state.pos[env_index]
        xform = rendering.Transform()
        xform.set_translation(*pos_region)
        region.add_attr(xform)

        return [region]
