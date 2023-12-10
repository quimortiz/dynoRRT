import sys  # noqa

sys.path.append(".")  # noqa

import os
import time
import numpy as np
import pydynorrt

cm = pydynorrt.Collision_manager_pinocchio()


base_path = os.environ["DYNORRT_PATH"]

urdf = base_path + "models/point_payload_two_robots.urdf"
srdf = base_path + "models/point_payload_two_robots.srdf"

# urdf = "/io/benchmark/models/point_payload_two_robots.urdf"
# srdf = "/io/benchmark/models/point_payload_two_robots.srdf"

# "/io/benchmark/models/point_payload_two_robots.urdf"

cm.set_urdf_filename(urdf)
cm.set_srdf_filename(srdf)
cm.build()


start = np.array([-0.62831853, 0.0, 0.0, 0.0, 0.9424778, 0.0, -0.9424778])

goal = np.array([0.62831853, 0.2, 0.3, 0.0, 0.9424778, 0.0, -0.9424778])

tic = time.time()
N = 1000
for i in range(N):
    o = cm.is_collision_free(start)
toc = time.time()
elapsed = toc - tic
print("Time of 1 Collision in ms: (including python overhead)", 1000.0 / N * elapsed)
print("Time Purely on C++", cm.get_time_ms() / N)

print("second time")
cm.reset_counters()
tic = time.time()
N = 1000
for i in range(N):
    o = cm.is_collision_free(start)
toc = time.time()
elapsed = toc - tic
print("Time of 1 Collision in ms: (including python overhead)", 1000.0 / N * elapsed)
print("Time Purely on C++", cm.get_time_ms() / N)


print("third time")
cm.reset_counters()
mid = (start + goal) / 2.0
tic = time.time()
N = 1000
for i in range(N):
    o = cm.is_collision_free(start)
toc = time.time()
elapsed = toc - tic
print("Time of 1 Collision in ms: (including python overhead)", 1000.0 / N * elapsed)
print("Time Purely on C++", cm.get_time_ms() / N)


assert cm.is_collision_free(goal)
assert not cm.is_collision_free((goal + start) / 2.0)
