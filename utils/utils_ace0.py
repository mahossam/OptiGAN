"""
Utility Functions
"""
from __future__ import print_function
from __future__ import division

import math
import numpy as np
import scipy.constants
from utils import states

__author__ = 'mikepsn, lrbenke'


def constrain_360(angle):
    """ Constrains the angle in degrees to (0, 360) """
    x = np.fmod(angle, 360)
    if x < 0:
        x += 360
    return x


def xx_constrain_180(angle):
    """ Constrains the angle in degrees to (-180, 180) """
    x = np.fmod(angle, 360)
    if angle > 180:
        angle -= 360
    return angle

def constrain_180(angle):
    """ Constrains the specified angle to +/- 180 deg. """
    a = angle
    a -= np.ceil(angle/360.0 - 0.5) * 360.0
    return a

def smallest_angle(a, b):
    """ Returns the smallest angle between two angles including direction """
    return min(b-a, b-a+360, b-a-360, key=abs)


def nautical_miles_to_metres(a):
    """ Converts nautical miles to metres """
    nautical_mile = scipy.constants.nautical_mile
    return a * nautical_mile


def metres_to_nautical_miles(a):
    """ Converts metres to nautical miles """
    nautical_mile = scipy.constants.nautical_mile
    return float(a)/nautical_mile


def feet_to_metres(a):
    """ Converts feet to metres """
    foot = scipy.constants.foot
    return a * foot


def metres_to_feet(a):
    """ Converts metres to feet """
    foot = scipy.constants.foot
    return float(a)/foot


def knots_to_mps(a):
    """ Converts knots to metres per second """
    knot = scipy.constants.knot
    return a * knot


def mps_to_knots(a):
    """ Converts metres per second to knots """
    knot = scipy.constants.knot
    return float(a)/knot


def mps_to_mach(v):
    """ Converts speed from metres per second to mach number. """
    mach = scipy.constants.mach
    return float(v)/mach


def distance(p1, p2):
    """ Calculates the distance between two positions """
    diff = np.subtract(p1, p2)
    return np.linalg.norm(diff)


def get_range(obj1, obj2):
    """ Calculates the range between two entities. """
    o1_loc = get_location_vector(obj1)
    o2_loc = get_location_vector(obj2)
    separation_vector = o1_loc - o2_loc
    return np.linalg.norm(separation_vector)


def get_location_vector(obj):
    """ Returns obj's location as [x, y, z]. """
    return np.matrix([obj.x, obj.y, obj.z])


def get_velocity_vector(obj):
    """ Calculates the velocity vector for obj. """
    return np.matrix([obj.v * np.cos(np.radians(obj.psi)),
                      obj.v * np.sin(np.radians(obj.psi)),
                      obj.v * np.sin(np.radians(obj.theta))])


def get_aspect_angle(obj1, obj2):
    """ Calculates the aspect angle between obj1 and obj2. """
    o1_loc = get_location_vector(obj1)
    o2_loc = get_location_vector(obj2)
    los_line = o1_loc - o2_loc
    o2_v = get_velocity_vector(obj2)
    cosine_taa = los_line * o2_v.transpose() / ((np.linalg.norm(los_line) * np.linalg.norm(o2_v)))
    return float(np.degrees(np.arccos(cosine_taa)))


def get_bearing_angle(obj1, obj2):
    """ Calculates bearing angle between obj1 and obj2. """
    o1_loc = get_location_vector(obj1)
    o2_loc = get_location_vector(obj2)
    separation_vector = o1_loc - o2_loc
    velocity_vector = get_velocity_vector(obj1)
    sv = separation_vector
    vv = velocity_vector.transpose() 
    vT = velocity_vector.transpose() 
    norm_sv = np.linalg.norm(sv)
    norm_vv = np.linalg.norm(vv)
    cosine_ba = sv * vT / (norm_sv * norm_vv)
    #cosine_ba = separation_vector * \
    # velocity_vector.transpose() / (#(np.linalg.norm(separation_vector) * np.linalg.norm(velocity_vector)))
    return float(np.degrees(np.arccos(cosine_ba)))


def get_antenna_train_angle(obj1, obj2):
    """ Calculates antenna train angle between obj1 and obj2. """
    o1_loc = get_location_vector(obj1)
    o2_loc = get_location_vector(obj2)
    los_line = o2_loc - o1_loc
    o1_v = get_velocity_vector(obj1)
    cosine_ata = los_line * o1_v.transpose() / ((np.linalg.norm(los_line) * np.linalg.norm(o1_v)))
    return float(np.degrees(np.arccos(cosine_ata)))


def relative_bearing(own_x, own_y, other_x, other_y):
    """ Calculates the bearing from one position to another """
    return constrain_360(
        np.degrees(np.arctan2(other_y - own_y, other_x - own_x)))


def reciprocal_heading(heading):
    """ Calculates the reciprocal (anti-parallel) heading """
    return constrain_360(heading + 180)


def is_reciprocal(h1, h2):
    """ Returns True if the headings are reciprocal (at 180 degrees) """
    relative_heading = constrain_360(h1 - h2)
    return is_close(relative_heading, 180.0)


def is_close(a, b, rel_tol=1e-09, abs_tol=0.0):
    """ Returns True if a and b are close to within floating point error (can
     be replaced by math.isclose() in Python 3.5+) """
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def lateral_displacement(own_x, own_y, threat_x, threat_y, 
                         threat_heading, threat_range):
    """ Returns the perpendicular distance to the threat flight path using the
    rule `sin(theta) = opposite/hypotenuse`; assumes headings in degrees """
    taa = target_aspect_angle(own_x, own_y, threat_x, threat_y, threat_heading)
    return abs(threat_range * math.sin(math.radians(taa)))


def target_aspect_angle(own_x, own_y, threat_x, threat_y, threat_heading):
    """ Calculates the target aspect angle (TAA) """
    threat_bearing = relative_bearing(own_x, own_y, threat_x, threat_y)
    return reciprocal_heading(threat_bearing) - threat_heading


def is_angle_ccw(angle_1, angle_2):
    """ Returns True if the shortest angle from angle_1 to angle_2 is
    counter-clockwise """
    diff = angle_2 - angle_1
    return diff > 180 if diff > 0 else diff >= -180

def calc_los_angle(x1, y1, x2, y2):
    """ Calculates the los angle between two points. """
    dy = y2 - y1
    dx = x2 - x1
    los = np.degrees(np.arctan2(dy, dx))
    return los 

def calc_ba_aa(los1, los2, psi1, psi2):
    """ Given the headings of two aircraft psi1 and psi2,
    as well as the los angles, los1 and los2, calculates
    the bearing angles ba1, ba2 and the aspect angles aa1 and aa2
    """
    _los1 = constrain_180(los1)
    _los2 = constrain_180(los2)
    _psi1 = constrain_180(psi1)
    _psi2 = constrain_180(psi2)
    
    #ba1 = constrain_180(_los - _psi1)
    #ba2 = constrain_180(_los - _psi2)
    ba1 = constrain_180(_psi1 - _los1)
    ba2 = constrain_180(_psi2 - _los2)

    if ba1 < 0:
        aa2 = constrain_180(-180.0 - ba1)
    else:
        aa2 = constrain_180(180.0 - ba1)

    if ba2 < 0:
        aa1 = constrain_180(-180.0 - ba2)
    else:
        aa1 = constrain_180(180.0 - ba2)

    #aa1 = constrain_180(180.0 - ba2)
    #aa2 = constrain_180(180.0 - ba1)
    #print("LOS, PSI = ", (_los, los, _psi1, psi1, _psi2, psi2, _psi1-_los,ba1))
    if ba1 < -180 or ba2 < -180:
        print("ANGLES = ", (los, _los, psi1, psi2, _psi1, _psi2, ba1, ba2, aa1, aa2))
    return (ba1, ba2, aa1, aa2)

def calc_mcgrew_angle(contact_aa, contact_ata):
    """ Returns the angular component of the mcgrew score. """
    aa = np.fabs(contact_aa)
    ata = np.fabs(contact_ata)
    sa = 0.5 * ((1 - float(aa)/180.0) + (1 - float(ata)/180.0)) 
    return float(sa)

def calc_mcgrew_range(contact_range):
    """ Returns the range component of the mcgrew score. """
    r = contact_range
    k = 5.0
    rd = 380.0
    sr = np.exp(-np.fabs(r - rd)/(k * 180.0))
    return float(sr)

def delta_v(v1, v2):
    """ Returns the speed difference between v2 and v1. """
    return v2 - v1

def lethal_v(delta_v):
    """ Returns true if the absolute speed difference between the 
        the two aircraft is less than 100 knots. """
    return mps_to_knots(np.fabs(delta_v)) <= 100.0

def inside_lethal_range(contact_range):
    """ Returns true if the contact range is within the minimum
    and maximim range of the gun. """
    gun_min_r = feet_to_metres(500.0)
    gun_max_r = feet_to_metres(3000.0)
    return gun_min_r <= contact_range <= gun_max_r

def inside_lethal_cone(contact_aa, contact_ata):
    """ Returns true if the contact is within the gun lethality cone. 
    """
    max_aa = 60.0
    max_ata = 30.0
    aa = np.fabs(contact_aa)
    ata = np.fabs(contact_ata)
    return aa <= max_aa and ata < max_ata

def viable_lethal_shot(contact_range, contact_aa, contact_ata):
    """ Returns true if we have a viable lethal shot which occurs
    if we are within the lethal range and the lethality cone. """
    within_lethal_range = inside_lethal_range(contact_range)
    within_lethal_cone = inside_lethal_cone(contact_aa, contact_ata)
    return within_lethal_range and within_lethal_cone

def shot_reward(viable_shot_i, viable_shot_j):
    """ If at anytime j has a viable shot against i, 
        then the reward should be -1. 
        If i has a viable shot against j then the reward should be +1.
        Otherwise the reward is 0. """
    shot_reward = 0

    if viable_shot_i:
        shot_reward = 1

    if viable_shot_j:
        shot_reward = -1

    return shot_reward


def xx_mcgrew_score(contact_range, contact_aa, contact_ata):
    """ DEPRECATED. """
    r = contact_range
    aa = contact_aa
    ata = contact_ata
    k = 0.001
    rd = 1000.0

    sa = 1 - ((1 - float(aa)/180.0) + (1 - float(ata)/180.0))
    sr = np.exp(-np.fabs(r - rd)/(k * 180.0))

    score = float(sa * sr)

    return score

def contact_assessment(my_state, contact_state):
    """ Assess the situation for the fighter's current opponent. """
    los1 = calc_los_angle(my_state.x, my_state.y,
                             contact_state.x, contact_state.y)
    los2 = calc_los_angle(contact_state.x, contact_state.y,
                             my_state.x, my_state.y)
    contact_range = float(get_range(my_state, contact_state))
    ba1, ba2, aa1, aa2 = calc_ba_aa(los1, los2, my_state.psi, contact_state.psi)
    contact_aa = aa1
    contact_ata = ba1
    mcgrew_angle = calc_mcgrew_angle(contact_aa, contact_ata)
    mcgrew_range = calc_mcgrew_range(contact_range)
    mcgrew_score = float(mcgrew_angle * mcgrew_range)
    delta_v_val = delta_v(my_state.v, contact_state.v)
    lethal_v_val = lethal_v(delta_v_val)
    lethal_range = inside_lethal_range(contact_range)
    lethal_cone = inside_lethal_cone(contact_aa, contact_ata)
    viable_lethal_shot_val = viable_lethal_shot(contact_range,
                                             contact_aa,
                                             contact_ata)
    opponent_viable_lethal_shot = viable_lethal_shot(contact_range,
                                                    aa2, ba2)
    reward = shot_reward(viable_lethal_shot_val,
                                 opponent_viable_lethal_shot)
    return mcgrew_score


# def build_state(fighter_df):
def build_state(fighter_state_array):
    """
    Takes a pandas dataframe and returns an ACE FighterState object
    """
    # s = states.FighterState(fighter_df['x'], fighter_df['y'], fighter_df['z'], None,
    #                         fighter_df['psi'], None, fighter_df['theta'], None,
    #                         fighter_df['phi'], None, fighter_df['v'], None,
    #                         fighter_df['gload'], None, None)
    # TODO, be carefull of the order here, must match the FighterState params, alwyas check the fighter_state_array from the caller
    s = states.FighterState(fighter_state_array[0], fighter_state_array[1], fighter_state_array[2], None,
                            fighter_state_array[3], None, fighter_state_array[4], None,
                            fighter_state_array[5], None, fighter_state_array[6], None,
                            fighter_state_array[7], None, None)
    return s