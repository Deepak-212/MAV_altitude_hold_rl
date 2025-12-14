import numpy as np
import model.aerosonde_parameters as MAV
from tools.rotations import quaternion_to_rotation, quaternion_to_euler

class MavDynamics:
    def __init__(self, Ts):
        self._ts_simulation = Ts
        # _state is the 13x1 internal state of the aircraft that is being propagated:
        # _state = [pn, pe, pd, u, v, w, e0, e1, e2, e3, p, q, r]
        self._state = np.array([[MAV.north0],  # (0)
                               [MAV.east0],   # (1)
                               [MAV.down0],   # (2)
                               [MAV.u0],    # (3)
                               [MAV.v0],    # (4)
                               [MAV.w0],    # (5)
                               [MAV.e0],    # (6)
                               [MAV.e1],    # (7)
                               [MAV.e2],    # (8)
                               [MAV.e3],    # (9)
                               [MAV.p0],    # (10)
                               [MAV.q0],    # (11)
                               [MAV.r0],    # (12)
                               ])
        self.gamma1 = MAV.gamma1
        self.gamma2 = MAV.gamma2
        self.gamma3 = MAV.gamma3
        self.gamma4 = MAV.gamma4
        self.gamma5 = MAV.gamma5
        self.gamma6 = MAV.gamma6
        self.gamma7 = MAV.gamma7
        self.gamma8 = MAV.gamma8
        self.Jx = MAV.Jx
        self.Jy = MAV.Jy
        self.Jz = MAV.Jz
        self.Jxz = MAV.Jxz

    ###################################
    # public functions
    def update(self, delta, wind=np.zeros((6,1))):
        '''
            Integrate the differential equations defining dynamics, update sensors
            delta = (delta_a, delta_e, delta_r, delta_t) are the control inputs
            wind is the wind vector in inertial coordinates
            wind: Wind vector [wn, we, wd, gust_u, gust_v, gust_w]
            Ts is the time step between function calls.
        '''
        self._update_velocity_data(wind)
        forces_moments = self._forces_moments(delta)
        self._rk4_step(forces_moments)
    
    def get_altitude(self): # this for rl 
        return -self._state.item(2)

    def external_set_state(self, new_state):
        self._state = new_state

    ###################################
    # private functions
    def _rk4_step(self, forces_moments):
        # Integrate ODE using Runge-Kutta RK4 algorithm
        time_step = self._ts_simulation
        k1 = self._f(self._state[0:13], forces_moments)
        k2 = self._f(self._state[0:13] + time_step/2.*k1, forces_moments)
        k3 = self._f(self._state[0:13] + time_step/2.*k2, forces_moments)
        k4 = self._f(self._state[0:13] + time_step*k3, forces_moments)
        self._state[0:13] += time_step/6 * (k1 + 2*k2 + 2*k3 + k4)

        # normalize the quaternion
        e0 = self._state.item(6)
        e1 = self._state.item(7)
        e2 = self._state.item(8)
        e3 = self._state.item(9)
        normE = np.sqrt(e0**2+e1**2+e2**2+e3**2)
        self._state[6][0] = self._state.item(6)/normE
        self._state[7][0] = self._state.item(7)/normE
        self._state[8][0] = self._state.item(8)/normE
        self._state[9][0] = self._state.item(9)/normE

    def _f(self, state, forces_moments):
       
        # Extract states
        pn = state.item(0)
        pe = state.item(1)
        pd = state.item(2)
        u = state.item(3)
        v = state.item(4)
        w = state.item(5)
        e0 = state.item(6)
        e1 = state.item(7)
        e2 = state.item(8)
        e3 = state.item(9)
        p = state.item(10)
        q = state.item(11)
        r = state.item(12)

        # Extract forces and moments
        fx = forces_moments.item(0)
        fy = forces_moments.item(1)
        fz = forces_moments.item(2)
        L = forces_moments.item(3)  # roll moment
        M = forces_moments.item(4)  # pitch moment
        N = forces_moments.item(5)  # yaw moment

        # Position Kinematics
        R_body_to_inertial = quaternion_to_rotation(np.array([[e0], [e1], [e2], [e3]]))
        V_inertial = R_body_to_inertial @ np.array([[u], [v], [w]])
        pn_dot = V_inertial.item(0)  # North velocity
        pe_dot = V_inertial.item(1)  # East velocity
        pd_dot = V_inertial.item(2)  # Down velocity (positive down)

        # Position Dynamics
        u_dot = r*v - q*w + fx/MAV.mass
        v_dot = p*w - r*u + fy/MAV.mass
        w_dot = q*u - p*v + fz/MAV.mass

        # rotational kinematics
        e0_dot = 0.5 * (-p*e1 - q*e2 - r*e3)
        e1_dot = 0.5 * (p*e0 + r*e2 - q*e3)
        e2_dot = 0.5 * (q*e0 - r*e1 + p*e3)
        e3_dot = 0.5 * (r*e0 + q*e1 - p*e2)


        # rotatonal dynamics
        p_dot = self.gamma1*p*q - self.gamma2*q*r + self.gamma3*L + self.gamma4*N
        q_dot = self.gamma5*p*r - self.gamma6*(p**2 - r**2) + (1.0/self.Jy)*M
        r_dot = self.gamma7*p*q - self.gamma1*q*r + self.gamma4*L + self.gamma8*N 
        
        x_dot = np.array([
            [pn_dot],
            [pe_dot],
            [pd_dot],
            [u_dot],
            [v_dot],
            [w_dot],
            [e0_dot],
            [e1_dot],
            [e2_dot],
            [e3_dot],
            [p_dot],
            [q_dot],
            [r_dot]
        ])
        
        return x_dot
    def _update_velocity_data(self, wind):
        # Rotation matrix Body -> Inertial
        quat = self._state[6:10]
        R_b_i = quaternion_to_rotation(quat)
        
        # Wind in Inertial Frame (Steady + Gust)
        # Assuming wind input is [wn, we, wd, gust_u, gust_v, gust_w]
        wind_steady_inertial = wind[0:3]
        
        # Transform wind to Body Frame
        # v_wind_body = R_i_b * v_wind_inertial
        self._wind = R_b_i.T @ wind_steady_inertial + wind[3:6] # Adding body-frame gust
        
        # Relative Velocity (Airspeed Vector)
        # V_r = V_ground - V_wind
        ur = self._state.item(3) - self._wind.item(0)
        vr = self._state.item(4) - self._wind.item(1)
        wr = self._state.item(5) - self._wind.item(2)
        
        # Compute Airspeed (Va), Alpha, Beta
        self._Va = np.sqrt(ur**2 + vr**2 + wr**2)
        
        if self._Va == 0:
            self._alpha = 0
            self._beta = 0
        else:
            self._alpha = np.arctan2(wr, ur)
            self._beta = np.arcsin(vr / self._Va)
    def _forces_moments(self, delta):
        """
        delta: [delta_a, delta_e, delta_r, delta_t] (aileron, elevator, rudder, throttle)
        """
        # Unpack State
        phi, theta, psi = 0, 0, 0 # Not strictly needed for F/M, used quaternions below
        p, q, r = self._state[10:13].flatten()
        
        # Unpack Delta (Control Surfaces)
        # Assuming delta passed as an object or list. 
        # If delta is an object (MsgDelta), use delta.aileron, etc.
        if hasattr(delta, 'aileron'):
            da, de, dr, dt = delta.aileron, delta.elevator, delta.rudder, delta.throttle
        else:
            da, de, dr, dt = delta[0], delta[1], delta[2], delta[3]

        # 1. GRAVITY
        quat = self._state[6:10]
        R_b_i = quaternion_to_rotation(quat)
        # F_gravity_body = R_b_i^T * [0, 0, mg]^T
        fg = R_b_i.T @ np.array([[0], [0], [MAV.mass * MAV.gravity]])
        fx_g, fy_g, fz_g = fg.flatten()

        # 2. PROPULSION
        # Motor model
        F_thrust, M_torque = self._motor_thrust_torque(self._Va, dt)
        
        # 3. AERODYNAMICS
        rho = MAV.rho
        S = MAV.S_wing
        b = MAV.b
        c = MAV.c
        
        # Sigmoid blending (if implementing high alpha stall) 
        # Linear Lift/Drag Models
        sigma_alpha = (1 + np.exp(-MAV.M * (self._alpha - MAV.alpha0))) + np.exp(MAV.M * (self._alpha + MAV.alpha0))
        sigma = (1 + np.exp(-MAV.M * (self._alpha - MAV.alpha0))) * (1 + np.exp(MAV.M * (self._alpha + MAV.alpha0))) / sigma_alpha

        CL = MAV.C_L_0 + MAV.C_L_alpha * self._alpha
        CD = MAV.C_D_0 + MAV.C_D_alpha * self._alpha # Simple linear drag
        # Note: Beard uses a blended CL/CD for high angles of attack. 

        # Forces in Stability Frame
        f_lift = 0.5 * rho * self._Va**2 * S * (CL + MAV.C_L_q * c / (2 * self._Va) * q + MAV.C_L_delta_e * de)
        f_drag = 0.5 * rho * self._Va**2 * S * (CD + MAV.C_D_q * c / (2 * self._Va) * q + MAV.C_D_delta_e * de)

        # Rotate from Stability Frame to Body Frame (by alpha)
        # fx_aero = -Drag * cos(alpha) + Lift * sin(alpha)
        # fz_aero = -Drag * sin(alpha) - Lift * cos(alpha)
        fx_aero = -f_drag * np.cos(self._alpha) + f_lift * np.sin(self._alpha)
        fz_aero = -f_drag * np.sin(self._alpha) - f_lift * np.cos(self._alpha)
        
        # Lateral Force (fy)
        fy_aero = 0.5 * rho * self._Va**2 * S * (MAV.C_Y_0 + MAV.C_Y_beta * self._beta + MAV.C_Y_p * b / (2 * self._Va) * p + MAV.C_Y_r * b / (2 * self._Va) * r + MAV.C_Y_delta_a * da + MAV.C_Y_delta_r * dr)

        # 4. MOMENTS
        # Roll Moment (l)
        l_aero = 0.5 * rho * self._Va**2 * S * b * (MAV.C_ell_0 + MAV.C_ell_beta * self._beta + MAV.C_ell_p * b / (2 * self._Va) * p + MAV.C_ell_r * b / (2 * self._Va) * r + MAV.C_ell_delta_a * da + MAV.C_ell_delta_r * dr)
        
        # Pitch Moment (m)
        m_aero = 0.5 * rho * self._Va**2 * S * c * (MAV.C_m_0 + MAV.C_m_alpha * self._alpha + MAV.C_m_q * c / (2 * self._Va) * q + MAV.C_m_delta_e * de)
        
        # Yaw Moment (n)
        n_aero = 0.5 * rho * self._Va**2 * S * b * (MAV.C_n_0 + MAV.C_n_beta * self._beta + MAV.C_n_p * b / (2 * self._Va) * p + MAV.C_n_r * b / (2 * self._Va) * r + MAV.C_n_delta_a * da + MAV.C_n_delta_r * dr)

        # 5. SUMMATION
        # Total Forces
        Fx = fx_g + fx_aero + F_thrust
        Fy = fy_g + fy_aero
        Fz = fz_g + fz_aero 
        
        # Total Moments
        # Note: Propeller torque acts in the opposite direction of rotation. 
        # Assuming prop rotates clockwise (standard), torque is negative roll (-Kt * omega^2)? 
        L_total = l_aero - M_torque 
        M_total = m_aero
        N_total = n_aero

        return np.array([[Fx], [Fy], [Fz], [L_total], [M_total], [N_total]])

    def _motor_thrust_torque(self, Va, delta_t):
        # Quadratic approximation from Beard & McLain
        # delta_t is throttle (0 to 1)
        
        # Voltage map
        v_in = MAV.V_max * delta_t
        
        # Propeller Speed calculation (Simplified Quadratic solution)
        # a*Omega^2 + b*Omega + c = 0
        # This is derived from matching motor torque to propeller torque
        a = MAV.C_Q0 * MAV.rho * np.power(MAV.D_prop, 5) / ((2.*np.pi)**2)
        b = (MAV.C_Q1 * MAV.rho * np.power(MAV.D_prop, 4) / (2.*np.pi)) * Va + MAV.KQ**2/MAV.R_motor
        c = MAV.C_Q2 * MAV.rho * np.power(MAV.D_prop, 3) * Va**2 - (MAV.KQ / MAV.R_motor) * v_in + MAV.KQ * MAV.i0
        
        # Solve for Omega (propeller angular speed)
        try:
            Omega_op = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
        except ValueError:
            Omega_op = 0 # Engine cutoff or negative solution
            
        # Compute Thrust
        # T = Ct0*rho*D^4*Omega^2/(4pi^2) + Ct1...
        # Using simplified Coeffs from params
        J_op = 2 * np.pi * Va / (Omega_op * MAV.D_prop)
        C_T = MAV.C_T0 + MAV.C_T1 * J_op + MAV.C_T2 * J_op**2
        C_Q = MAV.C_Q0 + MAV.C_Q1 * J_op + MAV.C_Q2 * J_op**2
        
        n = Omega_op / (2 * np.pi)
        T_p = MAV.rho * n**2 * np.power(MAV.D_prop, 4) * C_T
        Q_p = MAV.rho * n**2 * np.power(MAV.D_prop, 5) * C_Q
        
        return T_p, Q_p
