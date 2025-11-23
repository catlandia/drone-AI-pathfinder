"""Mission Planner for drone delivery operations.

Handles high-level mission planning including:
- Delivery selection and prioritization
- Route optimization
- Cost estimation
"""

import numpy as np
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum


class DeliveryPriority(Enum):
    """Priority levels for deliveries."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4


class DeliveryStatus(Enum):
    """Status of a delivery."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Delivery:
    """Represents a delivery task."""
    id: int
    pickup_location: np.ndarray
    dropoff_location: np.ndarray
    priority: DeliveryPriority = DeliveryPriority.MEDIUM
    status: DeliveryStatus = DeliveryStatus.PENDING
    weight: float = 1.0  # kg
    time_window: Optional[tuple] = None  # (start, end) in seconds

    def __post_init__(self):
        self.pickup_location = np.array(self.pickup_location, dtype=np.float32)
        self.dropoff_location = np.array(self.dropoff_location, dtype=np.float32)

    @property
    def distance(self) -> float:
        """Distance from pickup to dropoff."""
        return float(np.linalg.norm(self.dropoff_location - self.pickup_location))


class MissionPlanner:
    """Plans and manages drone delivery missions.

    Handles delivery selection, cost estimation, and route optimization.
    """

    # Cost factors for delivery estimation
    DISTANCE_COST_FACTOR = 1.0  # Cost per meter
    PRIORITY_COST_FACTOR = 0.5  # Bonus for high priority
    WEIGHT_COST_FACTOR = 0.2   # Penalty per kg
    TIME_COST_FACTOR = 0.1     # Penalty for time-sensitive deliveries

    def __init__(
        self,
        drone_position: np.ndarray = None,
        max_payload: float = 5.0,
        max_range: float = 50.0
    ):
        """Initialize mission planner.

        Args:
            drone_position: Current drone position
            max_payload: Maximum payload weight in kg
            max_range: Maximum flight range in meters
        """
        self.drone_position = np.array(drone_position if drone_position is not None else [0, 0, 1], dtype=np.float32)
        self.max_payload = max_payload
        self.max_range = max_range

        self.deliveries: List[Delivery] = []
        self.completed_deliveries: List[Delivery] = []
        self.current_delivery: Optional[Delivery] = None

        self._next_id = 0

    def add_delivery(
        self,
        pickup: np.ndarray,
        dropoff: np.ndarray,
        priority: DeliveryPriority = DeliveryPriority.MEDIUM,
        weight: float = 1.0,
        time_window: tuple = None
    ) -> Delivery:
        """Add a new delivery to the queue.

        Args:
            pickup: Pickup location [x, y, z]
            dropoff: Dropoff location [x, y, z]
            priority: Delivery priority
            weight: Package weight in kg
            time_window: Optional time window (start, end)

        Returns:
            Created Delivery object
        """
        delivery = Delivery(
            id=self._next_id,
            pickup_location=pickup,
            dropoff_location=dropoff,
            priority=priority,
            weight=weight,
            time_window=time_window
        )
        self._next_id += 1
        self.deliveries.append(delivery)
        return delivery

    def remove_delivery(self, delivery_id: int) -> bool:
        """Remove a delivery from the queue.

        Args:
            delivery_id: ID of delivery to remove

        Returns:
            True if removed, False if not found
        """
        for i, d in enumerate(self.deliveries):
            if d.id == delivery_id:
                self.deliveries.pop(i)
                return True
        return False

    def update_drone_position(self, position: np.ndarray):
        """Update current drone position."""
        self.drone_position = np.array(position, dtype=np.float32)

    def estimate_delivery_cost(self, delivery: Delivery) -> float:
        """Estimate cost/effort for a delivery.

        Lower cost = better choice.

        Args:
            delivery: Delivery to estimate

        Returns:
            Estimated cost value
        """
        # Distance from current position to pickup
        to_pickup = np.linalg.norm(delivery.pickup_location - self.drone_position)

        # Distance from pickup to dropoff
        pickup_to_dropoff = delivery.distance

        # Total distance
        total_distance = to_pickup + pickup_to_dropoff

        # Base cost from distance
        cost = total_distance * self.DISTANCE_COST_FACTOR

        # Priority bonus (lower cost for higher priority)
        priority_bonus = (5 - delivery.priority.value) * self.PRIORITY_COST_FACTOR * 10
        cost -= priority_bonus

        # Weight penalty
        cost += delivery.weight * self.WEIGHT_COST_FACTOR * 10

        # Time window urgency
        if delivery.time_window is not None:
            # Add penalty if time window is closing
            cost += self.TIME_COST_FACTOR * 50

        return max(0, cost)

    def select_next_delivery(self) -> Optional[Delivery]:
        """Select the best next delivery based on cost estimation.

        Returns:
            Best delivery to do next, or None if queue is empty
        """
        pending = [d for d in self.deliveries if d.status == DeliveryStatus.PENDING]

        if not pending:
            return None

        # Sort by estimated cost
        pending.sort(key=lambda d: self.estimate_delivery_cost(d))

        # Select best option
        best = pending[0]
        best.status = DeliveryStatus.IN_PROGRESS
        self.current_delivery = best

        return best

    def complete_delivery(self, delivery: Delivery, success: bool = True):
        """Mark a delivery as completed.

        Args:
            delivery: Delivery to complete
            success: Whether delivery was successful
        """
        delivery.status = DeliveryStatus.COMPLETED if success else DeliveryStatus.FAILED

        if delivery in self.deliveries:
            self.deliveries.remove(delivery)

        self.completed_deliveries.append(delivery)

        if self.current_delivery == delivery:
            self.current_delivery = None

    def get_route_waypoints(self, delivery: Delivery) -> List[np.ndarray]:
        """Generate waypoints for a delivery.

        Args:
            delivery: Delivery to plan route for

        Returns:
            List of waypoints [current -> pickup -> dropoff]
        """
        waypoints = []

        # Current position to pickup
        waypoints.append(self.drone_position.copy())
        waypoints.append(delivery.pickup_location.copy())

        # Pickup to dropoff
        waypoints.append(delivery.dropoff_location.copy())

        return waypoints

    def get_pending_count(self) -> int:
        """Get number of pending deliveries."""
        return len([d for d in self.deliveries if d.status == DeliveryStatus.PENDING])

    def get_statistics(self) -> Dict[str, Any]:
        """Get mission statistics.

        Returns:
            Dictionary with mission stats
        """
        completed = len(self.completed_deliveries)
        failed = len([d for d in self.completed_deliveries if d.status == DeliveryStatus.FAILED])
        pending = self.get_pending_count()

        total_distance = sum(d.distance for d in self.completed_deliveries)

        return {
            "completed": completed,
            "failed": failed,
            "pending": pending,
            "success_rate": (completed - failed) / max(completed, 1),
            "total_distance": total_distance,
            "in_progress": self.current_delivery is not None
        }


def generate_random_deliveries(
    num_deliveries: int,
    bounds: tuple,
    seed: int = None
) -> List[Delivery]:
    """Generate random deliveries for testing.

    Args:
        num_deliveries: Number of deliveries to generate
        bounds: (x_min, x_max, y_min, y_max, z_min, z_max)
        seed: Random seed

    Returns:
        List of Delivery objects
    """
    if seed is not None:
        np.random.seed(seed)

    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    deliveries = []

    for i in range(num_deliveries):
        pickup = np.array([
            np.random.uniform(x_min, x_max),
            np.random.uniform(y_min, y_max),
            np.random.uniform(max(z_min, 0.5), min(z_max, 3))
        ])

        dropoff = np.array([
            np.random.uniform(x_min, x_max),
            np.random.uniform(y_min, y_max),
            np.random.uniform(max(z_min, 0.5), min(z_max, 3))
        ])

        priority = np.random.choice(list(DeliveryPriority))
        weight = np.random.uniform(0.5, 4.0)

        delivery = Delivery(
            id=i,
            pickup_location=pickup,
            dropoff_location=dropoff,
            priority=priority,
            weight=weight
        )
        deliveries.append(delivery)

    return deliveries
