"""
zeus_model.py — single source of truth for the Zeus biped (8 actuated DOF;
10 in the ankle_roll what-if, see ZEUS_VARIANT).

Both the planner (Pinocchio, gaitopt env) and the plant (MuJoCo, base env) are
built from the SAME file, zeus_26-main/zeus_description/urdf/zeus.urdf, so the
two models cannot drift apart: no hand-shifted pelvis, no separate twin URDF.

Frame (robot, from clean_urdf.py): X forward, Y left, Z up, base_link at the
hip centre.  Mirror plane is Y = 0.

Joint order everywhere (library, env, policy) = the HARDWARE joint map
(zeus_link.nexus_proto.JOINT_NAMES, index = bus*4 + node-1):
    0 left_hip_pitch   1 left_hip_roll   2 left_knee_pitch   3 left_ankle_pitch
    4 right_hip_pitch  5 right_hip_roll  6 right_knee_pitch  7 right_ankle_pitch
so a trained policy's residual vector goes onto the wire without permutation.
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ZEUS_DESC = os.path.abspath(os.path.join(HERE, "..", "..", "..", "zeus_26-main",
                                         "zeus_description"))
SRC_URDF = os.path.join(ZEUS_DESC, "urdf", "zeus.urdf")
MESHDIR = os.path.join(ZEUS_DESC, "meshes")
CLEAN_CONFIG = os.path.join(ZEUS_DESC, "config", "zeus_model.yaml")

# ZEUS_VARIANT selects another robot of the same CAD family.  Unset = the real
# robot.  Every output of a variant lives in variants/<name>/, so the real
# robot's files are never touched.
#   ankle_roll   what-if: build_zeus.py adds an ankle-roll joint to zeus.urdf
#   mukesh       its own Fusion export WITH ankle roll (mukesh_description/,
#                cleaned by clean_urdf.py --config)
VARIANT = os.environ.get("ZEUS_VARIANT", "")
if VARIANT:
    VARIANT_DIR = os.path.join(HERE, "variants", VARIANT)
    URDF = os.path.join(VARIANT_DIR, f"zeus_{VARIANT}.urdf")
    PIN_URDF = os.path.join(VARIANT_DIR, "zeus_pin.urdf")
    XML = os.path.join(VARIANT_DIR, "zeus.xml")
else:
    URDF = SRC_URDF
    PIN_URDF = os.path.join(HERE, "zeus_pin.urdf")     # zeus.urdf + sole frames
    XML = os.path.join(HERE, "zeus.xml")

# Per-robot geometry the build and the planner need.
#   FOOT_L, FOOT_W   sole box (fore-aft, lateral) [m], from the foot mesh
#   GAP_PARTS        (part whose most-inboard vertex is the leg-gap point,
#                     body the Pinocchio frame hangs on) for each gap frame
FOOT_L, FOOT_W = 0.210, 0.104
GAP_PARTS = {"ankle_inner": ("ankle_motor", "knee_bracket"),
             "knee_inner": ("knee_bracket", "knee_bracket")}
if VARIANT == "mukesh":
    DESC = os.path.abspath(os.path.join(HERE, "..", "..", "..", "mukesh_description"))
    URDF = os.path.join(DESC, "urdf", "mukesh.urdf")
    MESHDIR = os.path.join(DESC, "meshes")
    CLEAN_CONFIG = os.path.join(DESC, "config", "mukesh_model.yaml")
    FOOT_L, FOOT_W = 0.161, 0.130
    # no separate brackets: the shank IS the ankle motor, the thigh the knee motor
    GAP_PARTS = {"ankle_inner": ("ankle_motor", "ankle_motor"),
                 "knee_inner": ("knee_motor", "knee_motor")}

_LEG = ["hip_pitch", "hip_roll", "knee_pitch", "ankle_pitch"]
if VARIANT in ("ankle_roll", "mukesh"):
    _LEG = _LEG + ["ankle_roll"]
JOINT_NAMES = [f"{s}_{j}" for s in ("left", "right") for j in _LEG]
NJ = len(JOINT_NAMES)
L_IDX, R_IDX = np.arange(0, NJ // 2), np.arange(NJ // 2, NJ)
SPRING_JOINTS = {"left_hip_pitch": "left_hip_pitch_spring",
                 "left_knee_pitch": "left_knee_pitch_spring",
                 "right_hip_pitch": "right_hip_pitch_spring",
                 "right_knee_pitch": "right_knee_pitch_spring"}
# hardware SPRING_NAMES order (nexus_proto): L hip, L knee, R hip, R knee
SPRING_ORDER = ["left_hip_pitch", "left_knee_pitch", "right_hip_pitch",
                "right_knee_pitch"]
SEA = {"left_hip_pitch", "left_knee_pitch", "right_hip_pitch", "right_knee_pitch"}
ROLL = {"left_hip_roll", "right_hip_roll"}
ANKLE = {"left_ankle_pitch", "right_ankle_pitch"}
ANKLE_ROLL = {"left_ankle_roll", "right_ankle_roll"} & set(JOINT_NAMES)
_idx = lambda group: np.array([i for i, n in enumerate(JOINT_NAMES) if n in group])
SEA_IDX = _idx(SEA)
ROLL_IDX = _idx(ROLL)
ANK_IDX = _idx(ANKLE)
AROLL_IDX = _idx(ANKLE_ROLL)
# Ankle-roll drive in the what-if: assumed the same class as the ankle pitch
# (small rotor, 15:1) until the sizing from its library says otherwise.
ANKLE_ROLL_MASS = float(os.environ.get("ANKLE_ROLL_MASS", 0.5))   # kg, at the ankle

# ── SEA spring (user spec 2026-10-06): 60 Nm/deg, NOT the old 170 Nm/deg ───
K_SPRING_NM_PER_DEG = float(os.environ.get("K_SPRING_DEG", 60.0))
K_SPRING = K_SPRING_NM_PER_DEG * 180.0 / np.pi          # 3437.7 Nm/rad
# Deflection budget (user spec 2026-10-25).  The spring is linear with no
# designed hard stop; it runs out of mechanical travel at about 1.6-1.7 deg.
#   SPRING_SOFT    1.0 deg   60 Nm  RL penalises deflection beyond this: keep
#                                   the SEA stiff and the recoil low
#   SPRING_PLAN    1.6 deg   96 Nm  planner hard limit on spring torque
#   SPRING_TRAVEL  1.7 deg  102 Nm  mechanical end, MuJoCo joint range
# Spring torque is NOT the motor torque: the rotor sits on the motor side, so
# tau_spring = u - J_rotor*qdd - D*qd (see gait_opt_zeus.py).
SPRING_SOFT = np.radians(float(os.environ.get("SPRING_SOFT_DEG", 1.0)))
SPRING_PLAN = np.radians(float(os.environ.get("SPRING_PLAN_DEG", 1.6)))
SPRING_TRAVEL = np.radians(float(os.environ.get("SPRING_TRAVEL_DEG", 1.7)))
TAU_SPRING_PLAN = K_SPRING * SPRING_PLAN                 # 96.0 Nm
SPRING_E_CAP = 0.5 * K_SPRING * SPRING_PLAN ** 2         # 1.340 J per spring
SPRING_DAMPING = 2.0                                     # Nm s/rad, small
SPRING_ARMATURE = 0.0002                                 # kg m^2, keeps the spring DOF well conditioned

# ── actuators: derived from motor + gear (geared build, 2026-09-21) ─────────
N_SEA, N_ROLL, N_ANK = 47.0, 26.0, 15.0
TAU_ROTOR_NOM_SEA, TAU_ROTOR_PEAK_SEA = 1.30, 4.00
TAU_ROTOR_NOM_SML, TAU_ROTOR_PEAK_SML = 0.58, 1.68
J_ROTOR_SEA = 1.1045 / N_SEA ** 2
_r = TAU_ROTOR_NOM_SML / TAU_ROTOR_NOM_SEA
J_ROTOR_SML = J_ROTOR_SEA * _r
C_ROTOR_SEA, C_ROTOR_SML = 2.0, 2.0 / _r ** 2
OMEGA_ROTOR_MAX = 6.4 * N_SEA
PEAK_DERATE = float(os.environ.get("PEAK_DERATE", 0.90))


def _by(sea, roll, ank):
    """Per-joint value by drive class; ankle roll takes the ankle class."""
    return np.array([sea if n in SEA else roll if n in ROLL else ank
                     for n in JOINT_NAMES], dtype=float)


GEAR = _by(N_SEA, N_ROLL, N_ANK)
ARMATURE = _by(J_ROTOR_SEA * N_SEA ** 2, J_ROTOR_SML * N_ROLL ** 2,
               J_ROTOR_SML * N_ANK ** 2)               # 1.1045 / 0.151 / 0.050
TAU_PEAK = _by(TAU_ROTOR_PEAK_SEA * N_SEA, TAU_ROTOR_PEAK_SML * N_ROLL,
               TAU_ROTOR_PEAK_SML * N_ANK)             # 188 / 43.7 / 25.2
TAU_LIMIT = TAU_PEAK * PEAK_DERATE                      # optimiser box
TAU_NOM = _by(TAU_ROTOR_NOM_SEA * N_SEA, TAU_ROTOR_NOM_SML * N_ROLL,
              TAU_ROTOR_NOM_SML * N_ANK)               # 61.1 / 15.1 / 8.7 RMS
VEL_LIMIT = _by(OMEGA_ROTOR_MAX / N_SEA, OMEGA_ROTOR_MAX / N_ROLL,
                OMEGA_ROTOR_MAX / N_ANK)
C_COPPER = _by(C_ROTOR_SEA / N_SEA ** 2, C_ROTOR_SML / N_ROLL ** 2,
               C_ROTOR_SML / N_ANK ** 2)
JOINT_DAMPING = _by(0.5, 0.5, 0.2)                      # output-side viscous
if len(AROLL_IDX):
    # Sizing runs set these generously, so the library shows what the gait
    # NEEDS rather than what an assumed drive allows.  Planner-only: TAU_PEAK
    # (the drive rating, written into zeus.xml) is left alone, so the built
    # model does not depend on how a library was sized.
    TAU_LIMIT[AROLL_IDX] = float(os.environ.get(
        "ANKLE_ROLL_PEAK", TAU_PEAK[AROLL_IDX][0])) * PEAK_DERATE
    TAU_NOM[AROLL_IDX] = float(os.environ.get("ANKLE_ROLL_NOM", TAU_NOM[AROLL_IDX][0]))

# position ranges for the optimiser (URDF hard limits are +-45 deg)
# Hand-set planning ranges (narrower than hardware where we want the optimiser
# to stay away from a limit).  These are then INTERSECTED with the model's own
# joint limits below — never trust a hand-set range to be inside the plant.
_OPT_RANGE_BY = {"hip_pitch": [-0.75, 0.75], "hip_roll": [-0.30, 0.30],
                 "knee_pitch": [-1.30, 1.30], "ankle_pitch": [-0.70, 0.70],
                 "ankle_roll": [-0.35, 0.35]}
_OPT_RANGE_WISH = np.array([_OPT_RANGE_BY[n.split("_", 1)[1]] for n in JOINT_NAMES])

# The knee wish (+-1.30 rad = +-74.5 deg) EXCEEDED the model's +-0.785 rad
# (+-45 deg) by 29.5 deg.  A reference planned outside the plant's limits is
# untrackable by construction: MuJoCo clamps it and the policy then burns
# torque chasing a pose that cannot exist.  Same class as the Pinocchio-vs-
# MuJoCo nv mismatch (docs/00_SPINE.md pattern P6).
# Resolution: take the model's limits as truth and keep a safety margin inside.
LIMIT_MARGIN = np.radians(float(os.environ.get("LIMIT_MARGIN_DEG", 2.0)))


def _model_joint_limits():
    """Joint limits straight from the compiled MuJoCo model."""
    import mujoco
    # read the ranges from the parsed spec: compiling the 494k-vertex meshes just to get
    # 10 numbers peaks at ~1 GB per process (angle="radian", so the values are identical)
    spec = mujoco.MjSpec.from_file(XML)
    lim = np.zeros((len(JOINT_NAMES), 2))
    for i, n in enumerate(JOINT_NAMES):
        lim[i] = spec.joint(n).range
    return lim


# Diagnostic escape hatch ONLY: lets a run use the hand-set ranges so the
# effect of clamping can be isolated.  A library built with this set is NOT
# trackable on hardware — the plant will clamp it.  Never ship one.
IGNORE_MODEL_LIMITS = os.environ.get("IGNORE_MODEL_LIMITS", "0") == "1"

try:
    if IGNORE_MODEL_LIMITS:
        raise RuntimeError("IGNORE_MODEL_LIMITS=1 (diagnostic only)")
    _MODEL_LIM = _model_joint_limits()
    OPT_RANGE = np.stack([
        np.maximum(_OPT_RANGE_WISH[:, 0], _MODEL_LIM[:, 0] + LIMIT_MARGIN),
        np.minimum(_OPT_RANGE_WISH[:, 1], _MODEL_LIM[:, 1] - LIMIT_MARGIN),
    ], axis=1)
    _clamped = [JOINT_NAMES[i] for i in range(len(JOINT_NAMES))
                if OPT_RANGE[i, 1] < _OPT_RANGE_WISH[i, 1] - 1e-9]
    if _clamped and os.environ.get("QUIET_LIMITS", "0") != "1":
        print(f"zeus_model: planning range CLAMPED to model limits "
              f"(margin {np.degrees(LIMIT_MARGIN):.1f} deg) for: "
              f"{', '.join(_clamped)}")
except Exception as _e:          # model not loadable -> fall back, but say so
    print(f"zeus_model: WARNING could not read model limits ({_e}); "
          f"using hand-set OPT_RANGE unchecked")
    OPT_RANGE = _OPT_RANGE_WISH

# Mirror about Y=0: left<->right.  Pitch joints all turn about +Y on both legs
# (clean_urdf.py), so a mirrored pitch keeps its sign; roll joints turn about
# +X and "positive moves the foot left" on both legs, so roll flips.  Verified
# numerically in validate_twin.py (sole reflection residual).
def mirror_matrix():
    S = np.zeros((NJ, NJ))
    for i, n in enumerate(JOINT_NAMES):
        side, joint = n.split("_", 1)
        j = JOINT_NAMES.index(("right_" if side == "left" else "left_") + joint)
        S[i, j] = -1.0 if joint.endswith("roll") else 1.0
    return S


MU = 0.7
GRAVITY = 9.81


def build_pin_model(lock_springs=True):
    """Free-flyer Pinocchio model of zeus_pin.urdf.  Springs locked at zero
    (the optimiser plans the output joint; the 3438 Nm/rad spring deflects
    <0.5 deg under walking loads).  Armature = reflected rotor inertia."""
    import pinocchio as pin
    full = pin.buildModelFromUrdf(PIN_URDF, pin.JointModelFreeFlyer())
    if lock_springs:
        lock = [full.getJointId(SPRING_JOINTS[n]) for n in SEA]
        model = pin.buildReducedModel(full, sorted(lock), pin.neutral(full))
    else:
        model = full
    arm = np.zeros(model.nv)
    damp = np.zeros(model.nv)
    for i, n in enumerate(JOINT_NAMES):
        iv = model.joints[model.getJointId(n)].idx_v
        arm[iv] = ARMATURE[i]
        damp[iv] = JOINT_DAMPING[i]
    if not lock_springs:
        for n in SPRING_JOINTS.values():
            arm[model.joints[model.getJointId(n)].idx_v] = SPRING_ARMATURE
    model.armature = arm
    return model, damp


def joint_idx(model):
    iq = np.array([model.joints[model.getJointId(n)].idx_q for n in JOINT_NAMES])
    iv = np.array([model.joints[model.getJointId(n)].idx_v for n in JOINT_NAMES])
    return iq, iv
