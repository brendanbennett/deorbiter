def compute_jacobian(self, state, dt):#Acutally just ignore this def, must have wrote it drunk
    # Extract position and velocity from the state vector
    x, y, vx, vy = state
    jacobian = np.zeros((4, 4))
    
    # Derivative of position w.r.t. velocity (identity multiplied by dt)
    jacobian[0, 2] = dt
    jacobian[1, 3] = dt
    
    # Derivative of velocity w.r.t. position
    r = np.sqrt(x**2 + y**2)
    grav_accel = -GM_EARTH / r**3
    jacobian[2, 0] = grav_accel * x * dt / r  # dvx/dx
    jacobian[3, 1] = grav_accel * y * dt / r  # dvy/dy
    
    #partial derivatives for velocity
    # Assuming the drag affects the velocity components?
    rho = self._atmosphere_model.density(state[:2], self.times[-1])  # Get density at current state
    drag_coeff = -0.5 * MEAN_DRAG_COEFF * MEAN_XSECTIONAL_AREA * rho / SATELLITE_MASS
    jacobian[2, 2] = (1 + drag_coeff * np.sqrt(vx**2 + vy**2) * vx) * dt  # dvx/dvx
    jacobian[3, 3] = (1 + drag_coeff * np.sqrt(vx**2 + vy**2) * vy) * dt  # dvy/dvy
    
    return jacobian
