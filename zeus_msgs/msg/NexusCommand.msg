# A correction to the stored gait, published on /zeus/command at ~250 Hz.
#
# The STM32 is already playing a reference trajectory; this is only the nudge
# on top of it. For every joint:
#
#     drive target = ref_angle + residual_rad
#
# Send all zeros with enable true and the robot walks the stored gait.
#
# UNITS: RADIANS on the output shaft, like every angle in NexusState. The USB
# protocol carries turns; zeus_link converts, so nothing on the ROS side ever
# has to. The STM32 rejects any |residual| above 0.1 turns = 0.6283 rad.
#
# Joint order is the J_* order in NexusState.msg.
#
# The board only moves while these keep arriving. Stop publishing and the last
# target is held for 200 ms, then the link fault idles every axis. After any
# fault, publish one message with enable false before enable true is accepted
# again - see the zeus_link README.

std_msgs/Header header

float32[8] residual_rad         # rad, added to NexusState.ref_angle
bool enable                     # false = stand down: idle the axes

# Soft start (protocol v11). With enable, HOLD arms into a hold instead of the
# gait: the board freezes the gait clock at its start phase, ramps every joint
# from where it stands to that phase's pose over 3 s, and holds there until a
# command arrives with hold false - then the stride starts. Without hold an arm
# still ramps, into the running gait. The residual is ignored until the gait
# runs. NexusState.stream_flags shows RAMP and HOLD while this happens.
bool hold

# Set both ankle drives' position to zero, once. Honoured only while the board
# is NOT armed. Setting it asserts that both feet are flat right now: the ankle
# drives come back from every reboot without a valid position.
bool zero_ankles

# Back to standing (protocol v12). With enable, during the ramp to the start
# pose or the hold there: ramp back over 3 s to the pose measured when the
# robot armed, and hold it, so a following enable false idles the drives with
# the legs where they started. Ignored once the gait runs. Final until disarm.
# NexusState.stream_flags shows RETURN (with RAMP while on the way).
bool return_home
