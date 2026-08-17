import os, sys, warnings, math
warnings.filterwarnings("ignore", category=DeprecationWarning)
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../common")
try:
    from participant import Game, Frame
except ImportError as err:
    print("player_template: 'participant' module cannot be imported:", err)
    raise
import helper
from action import ActionControl
from dataclasses import dataclass

# reset_reason
NONE = Game.NONE
GAME_START = Game.GAME_START
SCORE_MYTEAM = Game.SCORE_MYTEAM
SCORE_OPPONENT = Game.SCORE_OPPONENT
GAME_END = Game.GAME_END
DEADLOCK = Game.DEADLOCK
GOALKICK = Game.GOALKICK
CORNERKICK = Game.CORNERKICK
PENALTYKICK = Game.PENALTYKICK
HALFTIME = Game.HALFTIME
EPISODE_END = Game.EPISODE_END

# game_state
STATE_DEFAULT = Game.STATE_DEFAULT
STATE_KICKOFF = Game.STATE_KICKOFF
STATE_GOALKICK = Game.STATE_GOALKICK
STATE_CORNERKICK = Game.STATE_CORNERKICK
STATE_PENALTYKICK = Game.STATE_PENALTYKICK

# coordinates
MY_TEAM = Frame.MY_TEAM
OP_TEAM = Frame.OP_TEAM
BALL = Frame.BALL
X = Frame.X
Y = Frame.Y
Z = Frame.Z
TH = Frame.TH
ACTIVE = Frame.ACTIVE
TOUCH = Frame.TOUCH
BALL_POSSESSION = Frame.BALL_POSSESSION
GK_IDX, D1_IDX, D2_IDX, F1_IDX, F2_IDX = 0, 1, 2, 3, 4
K_MAX, K_MIN, DIST_EPS, FRAME_DT, FIELD_REF = 9.5, 5, 1e-6, 0.05, 10.14
BALL_DAMPING, BALL_FRICTION, BALL_STOP_SPEED, _BALL_G = 0.2, 0.30, 0.02, 9.81
ROBOT_MASS, ROBOT_TORQUE = (2.5, 2.0, 2.0, 1.5, 1.5), (0.8, 1.2, 1.2, 0.4, 0.4)
INTERCEPT_MAX_STEPS, INTERCEPT_EFF = 55, 0.82
CATCH_Z, AIR_Z = 0.30, 0.38
KO = {"active": False, "ours": False, "stage": 0, "frames": 0, "gk_had": False, "last": STATE_DEFAULT,
      "armed": False, "f2_kicked": False, "f2_had": False}
_IX_SMOOTH = {}
@dataclass
class ShootPlan:
    target_y: float
    kick_speed: float
    kick_angle: float
opp_goal_x = lambda field: field[X] * 0.5
own_goal_x = lambda field: -field[X] * 0.5
def _is_our_kickoff(cur_posture, cur_posture_opp, cur_ball):
    our_hold = any(cur_posture[i][BALL_POSSESSION] for i in range(5))
    opp_hold = any(cur_posture_opp[i][BALL_POSSESSION] for i in range(5))
    if opp_hold and not our_hold: return False
    if our_hold and not opp_hold: return True
    my_min = min(helper.distance(cur_posture[i][X], cur_ball[X], cur_posture[i][Y], cur_ball[Y]) for i in range(5))
    opp_min = min(helper.distance(cur_posture_opp[i][X], cur_ball[X], cur_posture_opp[i][Y], cur_ball[Y]) for i in range(5))
    return my_min + 0.20 < opp_min
def _our_has_ball(post): return any(p[BALL_POSSESSION] for p in post)
def kickoff_active(gs, post):
    if not KO["ours"] and _our_has_ball(post):
        KO["active"] = False
        return False
    return bool(KO["active"])
def sync_kickoff(gs, ball, post, post_opp=None, leader=False):
    if gs == STATE_KICKOFF and not KO.get("armed"):
        ours = True if post_opp is None else _is_our_kickoff(post, post_opp, ball)
        KO.update({"active": True, "ours": ours, "stage": 0, "frames": 0, "gk_had": False,
                   "armed": True, "f2_kicked": False, "f2_had": False})
    if gs != STATE_KICKOFF: KO["armed"] = False
    if not leader: return
    if KO["active"]:
        KO["frames"] += 1
        if post_opp is not None and any(p[BALL_POSSESSION] for p in post_opp):
            if not (KO["ours"] and KO.get("f2_kicked") and KO["stage"] < 3):
                if KO["ours"] and KO["stage"] >= 3 and not post[F2_IDX][BALL_POSSESSION]:
                    KO["active"] = False
                elif not KO["ours"]:
                    pass
                else:
                    KO["ours"] = False
        if KO["ours"]:
            if KO["f2_kicked"] and KO["stage"] < 1: KO["stage"] = 1
            if post[GK_IDX][BALL_POSSESSION]: KO["gk_had"], KO["stage"] = True, 2
            if KO["gk_had"] and not post[GK_IDX][BALL_POSSESSION] and KO["stage"] >= 2:
                KO["stage"] = 3
            if KO["stage"] >= 3 and post[F2_IDX][BALL_POSSESSION]:
                KO["f2_had"] = True
            if KO["stage"] >= 3 and KO["f2_had"] and not post[F2_IDX][BALL_POSSESSION]:
                KO["active"] = False
            elif KO["frames"] > 140:
                KO["active"] = False
        else:
            if _our_has_ball(post) or abs(ball[X]) > 0.55 or abs(ball[Y]) > 0.55 or KO["frames"] > 90: KO["active"] = False
            elif gs != STATE_KICKOFF and abs(ball[X]) > 0.25: KO["active"] = False
        if KO["frames"] > 135 or gs in (STATE_GOALKICK, STATE_CORNERKICK, STATE_PENALTYKICK): KO["active"] = False
        if gs != STATE_KICKOFF and not KO["active"]: KO["stage"] = 0
    elif gs != STATE_KICKOFF: KO["stage"] = 0
    KO["last"] = gs
def kickoff_action(self, robot_id, idx, cur_posture, cur_ball, prx, pry, cur_posture_opp=None, prev_ball=None):
    bx, by = cur_ball[X], cur_ball[Y]
    rx, ry = cur_posture[robot_id][X], cur_posture[robot_id][Y]
    ball_dis = helper.distance(rx, bx, ry, by)
    fh, fyh = self.field[X] / 2, self.field[Y] / 2
    spot_x, spot_y = fh * 0.32, min(1.05, fyh - 0.30)
    if prev_ball is None:
        prev_ball = cur_ball
    spd = ball_speed_per_frame(cur_ball, prev_ball)
    if not KO["ours"]:
        if robot_id == GK_IDX: return self.action.STOP()
        if robot_id == idx: return go_catch(self.action, prx, pry)
        holds = {D1_IDX: (-fh + 2.0, 0.0), D2_IDX: (-1.1, -1.15), F1_IDX: (-0.9, 1.15), F2_IDX: (-0.7, -0.35)}
        return self.action.go_to(*holds.get(robot_id, (-1.0, 0.0)))
    if robot_id == GK_IDX:
        if cur_posture[robot_id][BALL_POSSESSION] or ball_dis <= 0.22:
            if KO["frames"] < 25:
                return [0.2, 0.2, 0, 0, 0, 0]
            return [10, 10, 10, 10, 0, 0]
        return [0.2, 0.2, 0, 0, 0, 0]
    if robot_id == F2_IDX:
        if not KO["f2_kicked"]:
            behind = rx >= bx - 0.02
            if cur_posture[robot_id][BALL_POSSESSION] or (ball_dis <= 0.24 and behind):
                KO["f2_kicked"] = True
                return self.action.shoot_to(-fh + 0.12, 0.0, 9, 9)
            return self.action.go_to(bx + 0.20, by) if (not behind or ball_dis > 0.38) else self.action.go_to(bx, by)
        ball_flying = (KO["gk_had"] and not cur_posture[GK_IDX][BALL_POSSESSION]
                       and (spd > 0.03 or bx > cur_posture[GK_IDX][X] + 0.30))
        gk_out = KO["stage"] >= 3 or ball_flying
        if not gk_out:
            if math.hypot(spot_x - rx, spot_y - ry) > 0.30:
                return go_catch(self.action, spot_x, spot_y)
            return self.action.STOP()
        if cur_posture[robot_id][BALL_POSSESSION]:
            return self.action.shoot_to(fh, 0.0, 10, 10)
        return self.action.go_to(prx, pry)
    return self.action.go_to(fh - 2, 0)
def dribble_cx(robot_id, target_x, target_y, cur_posture, prev_posture, min_distance=0.65):
    rx, ry = cur_posture[robot_id][X], cur_posture[robot_id][Y]
    dx_t, dy_t = target_x - rx, target_y - ry
    dist_t = math.hypot(dx_t, dy_t)
    if dist_t < 0.01: return target_x, target_y
    tdx, tdy, lx, ly = dx_t / dist_t, dy_t / dist_t, -dy_t / dist_t, dx_t / dist_t
    den = dist_t if dist_t > 0.15 else 0.15
    lat, danger, max_proj, max_md = 0.0, False, 0.0, min_distance
    cands = [p for j, p in enumerate(cur_posture) if j != robot_id and p[BALL_POSSESSION]] or [p for j, p in enumerate(cur_posture) if j != robot_id]
    for m in cands:
        mx, my, held = m[X], m[Y], m[BALL_POSSESSION]
        md = min_distance * (1.5 if held else 1.0)
        inv, md2, bpm, om = 1.0 / (md + DIST_EPS), md * md, (2.0 if held else 1.0), (1.5 if held else 1.0)
        dx, dy = mx - rx, my - ry
        d2 = dx * dx + dy * dy
        if d2 > 2.0: continue
        dc, t = math.sqrt(d2), dx * tdx + dy * tdy
        if t <= 0.0: dpath, proj = dc, 0.0
        elif t >= dist_t: dpath, proj = math.hypot(rx + dist_t * tdx - mx, ry + dist_t * tdy - my), dist_t
        else: dpath, proj = abs(dx * tdy - dy * tdx), t
        if d2 < md2: dl = (md - dc) * inv * bpm
        elif dpath < md and proj < dist_t: dl = (md - dpath) * inv * (1.0 + 0.5 * (1.0 - proj / den)) * bpm
        else:
            dt = math.hypot(target_x - mx, target_y - my)
            dl = (md - dt) * inv * 0.5 * bpm if dt < md else 0.0
        if dl <= 0.1: continue
        danger, sgn = True, (1.0 if dx * lx + dy * ly > 0.0 else -1.0)
        if dpath < md: need = ((md - dpath) + 0.2) * sgn * om
        elif d2 < md2: need = ((md - dc) + 0.15) * sgn * om
        else: need = 0.0
        lat = max(-1.5, min(1.5, lat + need * dl))
        if proj < dist_t and dpath < md: max_proj, max_md = max(max_proj, proj), max(max_md, md)
    if not danger: return target_x, target_y
    prog = max(max_proj + max_md * 0.8, dist_t * 0.7) if 0.0 < max_proj < dist_t else dist_t
    pf = min(1.0, prog / dist_t)
    sx, sy = rx + tdx * prog * pf, ry + tdy * prog * pf
    if abs(lat) > 0.01:
        cx, cy = sx + lx * lat * (1.0 - pf * 0.5), sy + ly * lat * (1.0 - pf * 0.5)
        if not (cur_posture[robot_id][BALL_POSSESSION] and abs(ry) > 2.6 and abs(cy) > abs(ry)): sx, sy = cx, cy
    return sx, sy
def move_with_avoidance(action, robot_id, tx, ty, cur_posture, prev_posture, min_distance=0.65):
    return action.go_to(*dribble_cx(robot_id, tx, ty, cur_posture, prev_posture, min_distance))
def _shoot_ka(cur_ball, prev_ball, ry, field, mark_y=None):
    bx, by, fl, ogx = cur_ball[X], cur_ball[Y], field[X], field[X] * 0.5
    sc, hd = (fl / FIELD_REF if fl > 1e-6 else 1.0), math.hypot(bx - ogx, by - ry * 0.08)
    if bx < 0.0: h = K_MAX
    elif hd >= 5.0 * sc: h = 3.5 / max(hd - 3.8 * sc, 1e-3) + 6.33
    elif hd < 2.2 * sc: h = hd * 2.5
    else: h = hd * 2.2
    v1, v2 = helper.predict_ball_velocity(cur_ball, prev_ball, 1), helper.predict_ball_velocity(cur_ball, prev_ball, 2)
    bvm = math.hypot(v1[0] * 0.6 + v2[0] * 0.4, v1[1] * 0.6 + v2[1] * 0.4)
    if bvm > 0.1: h = max(4.0, min(float(K_MAX), h * (1.0 + bvm * 0.15)))
    if mark_y is not None and abs(by - mark_y) < 0.35: h = min(float(K_MAX), h * 1.06)
    return max(K_MIN, min(K_MAX, h))
def go_to_shoot(action, tx, ty, ks, ka):
    sp, rid = list(action.go_to(tx, ty)), action.robot_id
    p = action.cur_posture[rid]
    if p[BALL_POSSESSION]:
        d_th = abs(helper.wrap_to_pi(math.atan2(ty - p[Y], tx - p[X]) - p[TH]))
        if d_th < math.radians(20) or math.hypot(tx - p[X], ty - p[Y]) < 0.45: sp[2], sp[3] = ks, ka
    return sp
def _attack_control(self, robot_id, cur_ball, prev_ball, cur_posture, prev_posture, cur_posture_opp, idx_opp, defense_angle):
    action, field = self.action, self.field
    rx, ry, bx, by = cur_posture[robot_id][X], cur_posture[robot_id][Y], cur_ball[X], cur_ball[Y]
    fh, fyh, fl, fy = field[X] * 0.5, field[Y] * 0.5, field[X], field[Y]
    opp_x = cur_posture_opp[idx_opp][X] if idx_opp < len(cur_posture_opp) else 0.0
    opp_y = cur_posture_opp[idx_opp][Y] if idx_opp < len(cur_posture_opp) else 0.0
    opp_dist = math.hypot(rx - opp_x, ry - opp_y)
    self.d_pos = 1 if abs(ry) > max(1.0, 0.43 * fy) else 0
    look_goal = helper.looking_to_goal(cur_posture[robot_id], helper.relative_distance(fh, rx, 0, ry))
    v1, h = helper.predict_ball_velocity(cur_ball, prev_ball, 1), _shoot_ka(cur_ball, prev_ball, ry, field, mark_y=opp_y)
    abs_ty = 0.2 * bx if defense_angle > 0 else -0.2 * bx
    ty = abs_ty if abs(defense_angle) < 0.4 else (max(-0.15, min(0.15, v1[1] * 0.3)) * 1.3)
    mv = lambda tx, ty_: move_with_avoidance(action, robot_id, tx, ty_, cur_posture, prev_posture)
    if helper.distance(rx, field[X]/2, ry, 0) <= 1.9: return action.go_to(fh, ty)
    if opp_dist < 1.5 and bx < opp_x: return mv(bx + 0.5, ry + (0.8 if opp_y < ry else -0.8))
    if self.d_pos == 1 and abs(by) > 0.5:
        if bx > 0.345 * fl: self.d_pos = 0
        if bx > 0.1 * fl:
            if bx < 0.2 * fl and abs(by) < 0.55 * fy: return mv(0.3 * fl, fyh * (-1 if by < 0 else 1))
            return mv(fh, ty)
        yw, mid = (1 if by > 0 else -1), 0.6 * (fl / FIELD_REF)
        if -mid < bx < mid:
            return mv(fh, by) if abs(by) > 0.67 * fy else mv(fh - 0.2 * fl, 0.52 * fy * yw)
        return mv(0.2 * fl, 0.65 * yw) if bx <= -0.08 * fl else mv(fh - 0.05 * fl, ty)
    if bx > 0.1 * fl:
        self.time_count += 1
        align = max(8, int(20 - max(0.0, min(1.5, 1.85 - opp_dist)) * (12.0 / 1.5)))
        if self.time_count < align and not look_goal and bx < 0.375 * fl: return mv(fh, ty)
        return go_to_shoot(action, fh, ty, K_MAX if bx > 0.25 * fl else 0, h if bx > 0.25 * fl else 0)
    if abs(opp_y - by) < 0.6 and bx < opp_x: return mv(bx + 0.8, by * -1.35)
    return mv(bx + 0.8, by * 1.35) if abs(ry) <= 0.49 * fy else mv(0.5 * fl, by)
def _set_shootplan(cur_ball, prev_ball, defense_angle, field):
    ka = max(K_MIN, min(K_MAX, helper.distance(cur_ball[X], opp_goal_x(field), cur_ball[Y], 0) * 2.0 + 0.5))
    v1, bx = helper.predict_ball_velocity(cur_ball, prev_ball, 1), cur_ball[X]
    abs_ty = 0.2 * bx if defense_angle > 0 else -0.2 * bx
    ty = abs_ty if abs(defense_angle) < 0.4 else (max(-0.15, min(0.15, v1[1] * 0.3)) * 1.3)
    return ShootPlan(target_y=ty, kick_speed=K_MAX, kick_angle=ka)
def _should_tackle_opponent(cur_posture, cur_posture_opp, robot_id, idx_opp, robot_to_ball):
    rx, ry = cur_posture[robot_id][X], cur_posture[robot_id][Y]
    ox, oy = cur_posture_opp[idx_opp][X], cur_posture_opp[idx_opp][Y]
    return helper.distance(rx, ox, ry, oy) <= 0.2 and helper.looking_to_ball(cur_posture[robot_id], robot_to_ball) and cur_posture_opp[idx_opp][BALL_POSSESSION]
def ball_speed_per_frame(cur_ball, prev_ball):
    return math.hypot(cur_ball[X] - prev_ball[X], cur_ball[Y] - prev_ball[Y])
def ball_velocity_now(cur_ball, prev_ball):
    return (cur_ball[X] - prev_ball[X]) / FRAME_DT, (cur_ball[Y] - prev_ball[Y]) / FRAME_DT
def _robot_intercept_efficiency(robot_id):
    mass = ROBOT_MASS[robot_id] if 0 <= robot_id < len(ROBOT_MASS) else 2.0
    torque = ROBOT_TORQUE[robot_id] if 0 <= robot_id < len(ROBOT_TORQUE) else 0.8
    return INTERCEPT_EFF * (2.0 / mass) ** 0.25 * min(1.18, 0.95 + 0.25 * (torque / 1.2))
def simulate_ball_trajectory(cur_ball, prev_ball, max_steps):
    vx, vy = ball_velocity_now(cur_ball, prev_ball)
    x, y = cur_ball[X], cur_ball[Y]
    z = cur_ball[Z] if len(cur_ball) > 2 else 0.05
    vz = (cur_ball[Z] - prev_ball[Z]) / FRAME_DT if len(cur_ball) > 2 else 0.0
    decay, traj, stopped = math.exp(-BALL_DAMPING * FRAME_DT), [], False
    for _ in range(max_steps):
        if not stopped:
            vx *= decay; vy *= decay; sp = math.hypot(vx, vy)
            if sp > 1e-9:
                nsp = max(0.0, sp - BALL_FRICTION * FRAME_DT)
                if nsp < BALL_STOP_SPEED: stopped, vx, vy = True, 0.0, 0.0
                else:
                    k = nsp / sp; vx *= k; vy *= k
            else: stopped = True
            x += vx * FRAME_DT; y += vy * FRAME_DT
        vz -= _BALL_G * FRAME_DT; z = max(0.05, z + vz * FRAME_DT)
        if z <= 0.051 and vz < 0: vz *= -0.35
        traj.append((x, y, z))
    return traj
def compute_intercept_point(cur_ball, prev_ball, cur_posture, robot_id, max_lin_vel, max_steps=None, field=None):
    max_steps = INTERCEPT_MAX_STEPS if max_steps is None else max_steps
    rx, ry = cur_posture[robot_id][X], cur_posture[robot_id][Y]
    speed = ball_speed_per_frame(cur_ball, prev_ball)
    toward_own = (cur_ball[X] - prev_ball[X]) < -0.002
    step_reach = max_lin_vel * FRAME_DT * _robot_intercept_efficiency(robot_id) * (1.10 if toward_own else 1.0)
    slack = 0.12 + (0.08 if speed > 0.04 else 0.0) + (0.05 if toward_own else 0.0)
    raw = simulate_ball_trajectory(cur_ball, prev_ball, max_steps)
    if field is not None:
        hx, hy = field[X] / 2 - 0.12, field[Y] / 2 - 0.12
        traj = []
        for px, py, pz in raw:
            if abs(py) >= hy or abs(px) >= hx: break
            traj.append((px, py, pz))
        traj = traj or [(cur_ball[X], cur_ball[Y], cur_ball[Z] if len(cur_ball) > 2 else 0.05)]
    else: traj = raw
    for step, (px, py, pz) in enumerate(traj, start=1):
        if pz <= 0.40 and math.hypot(px - rx, py - ry) <= step_reach * step + slack: return px, py, step
    fx, fy, _ = traj[-1]
    return fx, fy, max(1, len(traj))
def _ball_traj_clipped(cur_ball, prev_ball, field, max_steps=None):
    max_steps = INTERCEPT_MAX_STEPS if max_steps is None else max_steps
    raw = simulate_ball_trajectory(cur_ball, prev_ball, max_steps)
    bz = cur_ball[Z] if len(cur_ball) > 2 else 0.05
    if field is None: return raw
    hx, hy = field[X] / 2 - 0.10, field[Y] / 2 - 0.10
    traj = []
    for px, py, pz in raw:
        if abs(px) >= hx or abs(py) >= hy: break
        traj.append((px, py, pz))
    return traj or [(cur_ball[X], cur_ball[Y], bz)]
def f2_receive_point(cur_ball, prev_ball, cur_posture, robot_id, max_v, field):
    rx, ry = cur_posture[robot_id][X], cur_posture[robot_id][Y]
    vx, vy = ball_velocity_now(cur_ball, prev_ball)
    spd = math.hypot(vx, vy)
    traj = _ball_traj_clipped(cur_ball, prev_ball, field, max_steps=80)
    step_r = max_v * FRAME_DT * _robot_intercept_efficiency(robot_id)
    land = None
    for step, (px, py, pz) in enumerate(traj, start=1):
        if pz > CATCH_Z: continue
        if land is None: land = (px, py, pz, step)
        slack = 0.22 + min(0.45, spd * FRAME_DT * 3.0)
        if math.hypot(px - rx, py - ry) <= step_r * step + slack:
            return px, py, pz, step
    if land is not None:
        return land[0], land[1], land[2], land[3]
    i = min(range(len(traj)), key=lambda k: traj[k][2])
    px, py, pz = traj[i]
    return px, py, pz, i + 1
def ball_pd(cur_ball, prev_ball, predicted_ball, cur_posture, robot_id, ball_dis, max_v=1.5, field=None):
    speed, bx, by = ball_speed_per_frame(cur_ball, prev_ball), cur_ball[X], cur_ball[Y]
    if speed < 0.006 and (len(cur_ball) < 3 or cur_ball[Z] < 0.18): px, py = bx, by
    else:
        px, py, _ = compute_intercept_point(cur_ball, prev_ball, cur_posture, robot_id, max_v, field=field)
        if ball_dis < 0.32: px, py = 0.4 * px + 0.6 * bx, 0.4 * py + 0.6 * by
    prev = _IX_SMOOTH.get(robot_id)
    if prev is not None: px, py = 0.72 * px + 0.28 * prev[0], 0.72 * py + 0.28 * prev[1]
    _IX_SMOOTH[robot_id] = (px, py)
    return px, py
def go_catch(action, x, y):
    sp, mv = action.go_to(x, y), action.max_linear_velocity[action.robot_id]
    lw, rw = sp[0], sp[1]
    if lw * rw < 0:
        sign = 1 if (lw + rw) >= 0 else -1
        a, b = (0.40, 0.95) if abs(rw) >= abs(lw) else (0.95, 0.40)
        return [*helper.set_wheel_velocity(mv, sign * mv * a, sign * mv * b), 0, 0, 0, 0]
    rid = action.robot_id
    d_th = abs(helper.wrap_to_pi(math.atan2(y - action.cur_posture[rid][Y], x - action.cur_posture[rid][X]) - action.cur_posture[rid][TH]))
    if d_th < helper.degree2radian(45):
        peak = max(abs(lw), abs(rw), 1e-6)
        if peak < mv * 0.90: lw, rw = helper.set_wheel_velocity(mv, lw * mv / peak, rw * mv / peak)
    return [lw, rw, 0, 0, 0, 0]
def apply_action(mode, robot_id, cur_posture, cur_posture_opp, idx, idx_opp, cur_ball, prev_ball,
                 defense_angle, d1x, d1y, robot_to_ball, prx, pry, self, prev_posture=None):
    gx, sp = opp_goal_x(self.field), _set_shootplan(cur_ball, prev_ball, defense_angle, self.field)
    ball_dis = helper.distance(cur_posture[robot_id][X], cur_ball[X], cur_posture[robot_id][Y], cur_ball[Y])
    sx, sy = cur_ball[X] - 0.5, cur_ball[Y] + (0.5 if cur_posture[robot_id][Y] < 0 else -0.5)
    if mode == "shoot": return self.action.shoot_to(gx, sp.target_y, sp.kick_speed if robot_id != D1_IDX else 10, sp.kick_angle if robot_id != D1_IDX else 10)
    if mode == "dribble": return _attack_control(self, robot_id, cur_ball, prev_ball, cur_posture, prev_posture, cur_posture_opp, idx_opp, defense_angle)
    self.time_count = 0
    if mode == "support": return self.action.go_to(sx, sy)
    if mode == "idle":
        if robot_id == D1_IDX:
            return go_catch(self.action, prx, pry) if ball_dis <= 0.6 else self.action.go_to(d1x, d1y)
        if cur_posture_opp[idx_opp][BALL_POSSESSION] and _should_tackle_opponent(cur_posture, cur_posture_opp, robot_id, idx_opp, robot_to_ball):
            return self.action.SLIDE()
        if idx == robot_id or cur_posture_opp[idx_opp][BALL_POSSESSION]: return go_catch(self.action, prx, pry)
        return self.action.go_to(sx, sy)
    return go_catch(self.action, prx, pry)
def _player_common(self, field, goal, penalty_area, goal_area, robot_size, max_linear_velocity, robot_id):
    self.field, self.goal, self.penalty_area, self.goal_area = field, goal, penalty_area, goal_area
    self.robot_size, self.max_linear_velocity = robot_size, max_linear_velocity
    self.action = ActionControl(robot_id, max_linear_velocity)
    self.gk_index, self.d1_index, self.d2_index, self.f1_index, self.f2_index = GK_IDX, D1_IDX, D2_IDX, F1_IDX, F2_IDX
    self.d_pos = self.time_count = 0
def _field_move(self, robot_id, idx, idx_opp, defense_angle, cur_posture, cur_posture_opp,
                prev_posture, prev_ball, cur_ball, predicted_ball, game_state, possess_mode):
    ball_dis = helper.distance(cur_posture[robot_id][X], cur_ball[X], cur_posture[robot_id][Y], cur_ball[Y])
    robot_to_ball = helper.relative_distance(cur_ball[X], cur_posture[robot_id][X], cur_ball[Y], cur_posture[robot_id][Y])
    prx, pry = ball_pd(cur_ball, prev_ball, predicted_ball, cur_posture, robot_id, ball_dis, self.max_linear_velocity[robot_id], self.field)
    self.action.update_state(cur_posture, prev_posture, cur_ball, prev_ball)
    sync_kickoff(game_state, cur_ball, cur_posture, cur_posture_opp)
    if kickoff_active(game_state, cur_posture):
        return kickoff_action(self, robot_id, idx, cur_posture, cur_ball, prx, pry, cur_posture_opp, prev_ball)
    mode = possess_mode if cur_posture[robot_id][BALL_POSSESSION] else "idle"
    if robot_id == D1_IDX:
        r = min(2.2, max(0.9, 1.8 * (self.field[X] / FIELD_REF)))
        d1x, d1y = math.cos(defense_angle) * r + own_goal_x(self.field), math.sin(defense_angle) * r
        return apply_action(mode, robot_id, cur_posture, cur_posture_opp, idx, idx_opp, cur_ball, prev_ball,
                            defense_angle, d1x, d1y, robot_to_ball, prx, pry, self, prev_posture)
    return apply_action(mode, robot_id, cur_posture, cur_posture_opp, idx, idx_opp, cur_ball, prev_ball,
                        defense_angle, prx, pry, robot_to_ball, prx, pry, self, prev_posture)
class Goalkeeper:
    def __init__(self, field, goal, penalty_area, goal_area, robot_size, max_linear_velocity, robot_id=0):
        _player_common(self, field, goal, penalty_area, goal_area, robot_size, max_linear_velocity, robot_id)
    def move(self, robot_id, idx, idx_opp, defense_angle, attack_angle, cur_posture, cur_posture_opp,
             prev_posture, prev_posture_opp, prev_ball, cur_ball, predicted_ball, game_state, target=[0, 0]):
        self.action.update_state(cur_posture, prev_posture, cur_ball, prev_ball)
        sync_kickoff(game_state, cur_ball, cur_posture, cur_posture_opp, leader=True)
        home_x = -self.field[X] / 2
        if kickoff_active(game_state, cur_posture):
            return kickoff_action(self, robot_id, idx, cur_posture, cur_ball, home_x, 0, cur_posture_opp, prev_ball)
        if game_state == STATE_GOALKICK: return [1, 1, 10, 10, 0, 0]
        if game_state == STATE_PENALTYKICK:
            if cur_ball[Y] > 0.001: return [0, 0, 0, 0, 10, 0]
            if cur_ball[Y] < -0.001: return [0, 0, 0, 0, 6, 0]
            return [0, 0, 0, 0, 0, 0]
        if cur_posture[robot_id][BALL_POSSESSION]: return self.action.shoot_to(self.field[X] / 2, 0, 10, 10)
        speeds = self.action.defend_ball()
        if speeds is not None: return speeds
        gx, gy = cur_posture[robot_id][X], cur_posture[robot_id][Y]
        if home_x - 0.05 < gx < home_x + 0.15 and abs(gy) < 0.05: return self.action.turn_to(0, 0)
        return self.action.go_to(home_x, 0)

class Defender_1:
    def __init__(self, field, goal, penalty_area, goal_area, robot_size, max_linear_velocity, robot_id=0):
        _player_common(self, field, goal, penalty_area, goal_area, robot_size, max_linear_velocity, robot_id)
    def move(self, robot_id, idx, idx_opp, defense_angle, attack_angle, cur_posture, cur_posture_opp,
             prev_posture, prev_posture_opp, prev_ball, cur_ball, predicted_ball, game_state, target=[0, 0]):
        return _field_move(self, robot_id, idx, idx_opp, defense_angle, cur_posture, cur_posture_opp,
                           prev_posture, prev_ball, cur_ball, predicted_ball, game_state, "shoot")

class Defender_2:
    def __init__(self, field, goal, penalty_area, goal_area, robot_size, max_linear_velocity, robot_id=0):
        _player_common(self, field, goal, penalty_area, goal_area, robot_size, max_linear_velocity, robot_id)
    def move(self, robot_id, idx, idx_opp, defense_angle, attack_angle, cur_posture, cur_posture_opp,
             prev_posture, prev_posture_opp, prev_ball, cur_ball, predicted_ball, game_state, target=[0, 0]):
        return _field_move(self, robot_id, idx, idx_opp, defense_angle, cur_posture, cur_posture_opp,
                           prev_posture, prev_ball, cur_ball, predicted_ball, game_state, "dribble")

class Forward_1:
    def __init__(self, field, goal, penalty_area, goal_area, robot_size, max_linear_velocity, robot_id=0):
        _player_common(self, field, goal, penalty_area, goal_area, robot_size, max_linear_velocity, robot_id)
    def move(self, robot_id, idx, idx_opp, defense_angle, attack_angle, cur_posture, cur_posture_opp,
             prev_posture, prev_posture_opp, prev_ball, cur_ball, predicted_ball, game_state, target=[0, 0]):
        if game_state == STATE_CORNERKICK and cur_ball[X] >= 0: return [5, 5, 0, 0, 0, 0]
        return _field_move(self, robot_id, idx, idx_opp, defense_angle, cur_posture, cur_posture_opp,
                           prev_posture, prev_ball, cur_ball, predicted_ball, game_state, "dribble")

class Forward_2:
    def __init__(self, field, goal, penalty_area, goal_area, robot_size, max_linear_velocity, robot_id=0):
        _player_common(self, field, goal, penalty_area, goal_area, robot_size, max_linear_velocity, robot_id)
    def move(self, robot_id, idx, idx_opp, defense_angle, attack_angle, cur_posture, cur_posture_opp,
             prev_posture, prev_posture_opp, prev_ball, cur_ball, predicted_ball, game_state, target=[0, 0]):
        ball_dis = helper.distance(cur_ball[X], cur_posture[robot_id][X], cur_ball[Y], cur_posture[robot_id][Y])
        if game_state == STATE_PENALTYKICK: return [0, 1, 10, 6, 0, 0] if ball_dis <= 0.2 else [0.8, 0.8, 0, 0, 0, 0]
        if game_state == STATE_CORNERKICK: return [1, 1, 10, 1, 0, 0]
        if game_state == STATE_GOALKICK:
            prx, pry = ball_pd(cur_ball, prev_ball, predicted_ball, cur_posture, robot_id, ball_dis, self.max_linear_velocity[robot_id], self.field)
            self.action.update_state(cur_posture, prev_posture, cur_ball, prev_ball)
            return go_catch(self.action, prx, pry)
        return _field_move(self, robot_id, idx, idx_opp, defense_angle, cur_posture, cur_posture_opp,
                           prev_posture, prev_ball, cur_ball, predicted_ball, game_state, "dribble")
    
