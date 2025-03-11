from datetime import datetime, timedelta
from collections import defaultdict, deque

from utils.logger import get_logger

class Person:
    """
    Represents a tracked person instance.

    This class maintains a unique instance for each tracking ID, storing the person's bounding box history,
    first seen time, last seen time, and providing utilities for tracking and cleanup.
    """
    _instances = {}

    def __new__(cls, tracking_id, *args, **kwargs):
        """
        Ensures that a single instance exists for each tracking ID.

        Args:
            tracking_id (int): Unique identifier for the tracked person.

        Returns:
            Person: An instance of the `Person` class.
        """
        if tracking_id not in cls._instances:
            cls._instances[tracking_id] = super(Person, cls).__new__(cls)
        return cls._instances[tracking_id]

    def __init__(self, tracking_id, max_bbox_history=100):
        """
        Initializes a new `Person` instance or reuses an existing one.

        Args:
            tracking_id (int): Unique identifier for the tracked person.
            max_bbox_history (int): Maximum number of bounding boxes to retain in history.
        """
        if not hasattr(self, '_initialized'):
            self.tracking_id = tracking_id
            self.bbox_history = deque(maxlen=max_bbox_history)
            self.first_seen_time = datetime.now()
            self.last_seen_time = self.first_seen_time
            self._initialized = True

    @property
    def life_span(self):
        """
        Computes the total lifespan of the person being tracked.

        Returns:
            timedelta: Duration between first seen and last seen times.
        """
        return self.last_seen_time - self.first_seen_time

    def update(self, bbox):
        """
        Updates the person's bounding box history and last seen time.

        Args:
            bbox (list or tuple): Bounding box coordinates [x1, y1, x2, y2].
        """
        self.bbox_history.append(bbox)
        self.last_seen_time = datetime.now()

    def get_recent_bboxes(self, count=1):
        """
        Retrieves the most recent bounding boxes.

        Args:
            count (int): Number of recent bounding boxes to retrieve.

        Returns:
            list: List of bounding boxes, up to `count` in reverse order (most recent first).
        """
        count = min(count, len(self.bbox_history))
        return list(self.bbox_history)[-count:][::-1]

    @classmethod
    def clean_up(cls, timeout=timedelta(seconds=30), verbose=False):
        """
        Cleans up stale instances of `Person` not updated within the timeout duration.
        
        Args:
            timeout (timedelta): Duration after which to remove stale instances.
            verbose (bool): If True, logs the IDs of removed instances.
        
        Returns:
            list: List of removed tracking IDs.
        """
        now = datetime.now()
        to_remove = [
            tracking_id for tracking_id, person in cls._instances.items()
            if now - person.last_seen_time > timeout
        ]
        for tracking_id in to_remove:
            if verbose:
                get_logger().debug(f"Removed ID: {tracking_id} /  {cls._instances[tracking_id]}")
            del cls._instances[tracking_id]
        return to_remove

    @classmethod
    def get_instances(cls):
        """
        Retrieves all active `Person` instances.

        Returns:
            dict: Dictionary mapping tracking IDs to their respective `Person` instances.
        """
        return cls._instances

    def __repr__(self):
        return (f"First seen: {self.first_seen_time.strftime('%Y-%m-%d %H:%M:%S')}, "
                f"Last seen: {self.last_seen_time.strftime('%Y-%m-%d %H:%M:%S')}, "
                f"Last box: {self.get_recent_bboxes(1)}")


if __name__ == '__main__':
    from datetime import timedelta
    from time import sleep

    # Step 1: Initialize or retrieve a `Person` instance
    tracking_id_1 = 101
    tracking_id_2 = 102

    person1 = Person(tracking_id_1)
    person2 = Person(tracking_id_2)

    # Step 2: Update bounding box history for tracked persons
    # Assume bounding boxes are in the format [x1, y1, x2, y2]
    person1.update([50, 50, 100, 100])
    person2.update([200, 200, 250, 250])
    sleep(5)
    person1.update([55, 55, 105, 105])


    # Step 3: Retrieve recent bounding box history
    recent_bboxes_person1 = person1.get_recent_bboxes(count=2)
    print(f"Recent bounding boxes for Person {tracking_id_1}: {recent_bboxes_person1}")

    recent_bboxes_person2 = person2.get_recent_bboxes(count=1)
    print(f"Recent bounding boxes for Person {tracking_id_2}: {recent_bboxes_person2}")

    # Step 4: Access lifespan and other properties
    print(f"Person {tracking_id_1} first seen: {person1.first_seen_time}")
    print(f"Person {tracking_id_1} last seen: {person1.last_seen_time}")
    print(f"Person {tracking_id_1} lifespan: {person1.life_span}")

    # Step 5: Clean up stale instances
    # Simulate some time passing
    sleep(27)  # Wait 27 seconds to simulate timeout
    removed_ids = Person.clean_up(timeout=timedelta(seconds=30))
    print(f"Removed tracking IDs: {removed_ids}")

    # Step 6: Check active instances
    active_instances = Person.get_instances()
    print(f"Active tracking: {active_instances}")

    # Step 7: Create a new `Person` instance after cleanup
    new_person = Person(103)
    new_person.update([300, 300, 350, 350])
    print(f"New person with tracking ID 103 added. Lifespan: {new_person.life_span}")

    # Expected Output:
    # Recent bounding boxes for Person 101: [[55, 55, 105, 105], [50, 50, 100, 100]]
    # Recent bounding boxes for Person 102: [[200, 200, 250, 250]]
    # Person 101 first seen: <timestamp>
    # Person 101 last seen: <timestamp>
    # Person 101 lifespan: <time difference>
    # Removed tracking IDs: [101, 102]  # Assuming both are stale
    # Active tracking IDs: [103]