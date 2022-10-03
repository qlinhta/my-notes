'''
Given the radius and the position of the center of a circle, implement the function randPoint which generates a uniform
random point inside the circle.

Implement the Solution class:

- Solution(double radius, double x_center, double y_center) initializes the object with the radius of the circle radius
and the position of the center (x_center, y_center).
- randPoint() returns a random point inside the circle. A point on the circumference of the circle is considered to be
in the circle. The answer is returned as an array [x, y].
'''

import random
import math

class Solution:

        def __init__(self, radius: float, x_center: float, y_center: float):
            self.radius = radius
            self.x_center = x_center
            self.y_center = y_center

        def randPoint(self) -> list[float]:
            # Generate random angle
            angle = random.uniform(0, 2*math.pi)
            # Generate random radius
            r = math.sqrt(random.uniform(0, 1)) * self.radius
            # Calculate x and y
            x = self.x_center + r * math.cos(angle)
            y = self.y_center + r * math.sin(angle)
            return [x, y]

if __name__ == "__main__":
    obj = Solution(1, 0, 0)
    param_1 = obj.randPoint()
    print(param_1)
