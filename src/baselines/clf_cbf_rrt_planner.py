"""
C-CLF-CBF-RRT Planner for Unicycle Model

Implements a Rapidly-exploring Random Tree (RRT) planner that uses Control Lyapunov
Functions (CLF) and Control Barrier Functions (CBF) to ensure both goal-reaching
and safety properties. The planner solves a Quadratic Program (QP) at each step to
check CLF-CBF compatibility.

Based on the framework from:
- Ames et al. "Control Barrier Function Based Quadratic Programs for Safety Critical Systems"
- Jankovic et al. "Robust Control Barrier Functions for Constrained Stabilization of Nonlinear Systems"
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

from src.environment.warehouse import WarehouseEnv


@dataclass
class RRTNode:
    """Node in the RRT tree."""
    state: np.ndarray  # [x, y, theta, v] - internal state
    parent: Optional['RRTNode'] = None
    cost: float = 0.0  # Cost from root to this node


class UnicyclePlannerCore:
    """
    Core planner that handles Lie derivative calculations and QP solving
    for CLF-CBF compatibility checking.
    """
    
    def __init__(self, env: WarehouseEnv, config: Dict[str, Any]):
        self.env = env
        self.config = config
        self.v_max = env.v_max
        self.a_max = env.a_max
        self.omega_max = env.omega_max
        self.robot_radius = env.robot_radius
        self.dt = env.dt
        
        # CLF Parameters
        self.P_clf = np.diag([10.0, 10.0, 0.1, 1.0])  # [x, y, theta, v] weights
        self.alpha_clf = config['train'].get('alpha_clf', 1.0)  # decay rate
        
        # CBF Parameters (Relative Degree 2)
        self.gamma_cbf = config['train'].get('gamma_cbf', 1.5)  # for B = h_dot + gamma*h
        self.alpha_cbf = config['train'].get('alpha_cbf', 1.5)  # for B_dot + alpha*B >= 0
        
        # Numerical differentiation step size
        self.epsilon = 1e-4
        
    def _get_f_g(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns f(x) and g(x) for the unicycle model.
        
        Unicycle dynamics:
        dx/dt = v * cos(theta)
        dy/dt = v * sin(theta)
        dtheta/dt = omega
        dv/dt = a
        
        So f(x) = [v*cos(theta), v*sin(theta), 0, 0]^T
        and g(x) = [[0, 0], [0, 0], [0, 1], [1, 0]]^T
        """
        _, _, theta, v = x
        f = np.array([v * np.cos(theta), v * np.sin(theta), 0.0, 0.0])
        g = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        return f, g

    def _get_clf_derivatives(self, x: np.ndarray, q: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Calculates CLF derivatives for the QP constraint.
        
        CLF: V(x) = (x - q)^T P (x - q)
        CLF condition: LfV + LgV * u <= -alpha_clf * V
        
        Returns:
            A_clf: 1x2 array, b_clf: scalar
            Constraint: A_clf @ u <= b_clf
        """
        # Ensure q is 4D (extend with theta=0, v=0 if only 2D position given)
        if len(q) == 2:
            q_full = np.array([q[0], q[1], 0.0, 0.0])
        else:
            q_full = q.copy()
            
        x_minus_q = x - q_full
        V = x_minus_q.T @ self.P_clf @ x_minus_q
        
        # grad(V) = 2 * P_clf * (x - q)
        grad_V = 2 * self.P_clf @ x_minus_q
        
        f, g = self._get_f_g(x)
        LfV = grad_V.T @ f
        LgV = grad_V.T @ g  # This is a 1x2 row vector: [LgV_a, LgV_omega]
        
        # CLF condition: LgV * u <= -LfV - alpha_clf * V
        # When far from goal and stopped, this can be too strict.
        # Use adaptive alpha: smaller when far, larger when close
        pos_dist = np.linalg.norm(x[:2] - q_full[:2])
        adaptive_alpha = self.alpha_clf * min(1.0, 2.0 / max(pos_dist, 0.1))  # Scale down when far
        
        b_clf = -LfV - adaptive_alpha * V
        
        return LgV.reshape(1, 2), b_clf.item()

    def _get_cbf_derivatives(self, x: np.ndarray, obs) -> Tuple[float, float, np.ndarray, float]:
        """
        Calculates CBF derivatives for a single obstacle.
        
        For relative degree 2 CBF, we need:
        - h: signed distance function
        - LfH: first Lie derivative
        - LgLfH: second Lie derivative (coefficient of u)
        - Lf^2H: second Lie derivative (control-independent term)
        
        Returns:
            h: scalar, LfH: scalar, LgLfH: 1x2 array, Lf2H: scalar
        """
        p = x[:2]  # position [x, y]
        theta, v = x[2], x[3]
        
        # Get signed distance and its gradient
        h = obs.get_signed_distance(p) - self.robot_radius
        
        # Compute gradient of h with respect to position using numerical differentiation
        grad_h = np.zeros(2)
        for i in range(2):
            p_plus = p.copy()
            p_plus[i] += self.epsilon
            p_minus = p.copy()
            p_minus[i] -= self.epsilon
            
            h_plus = obs.get_signed_distance(p_plus) - self.robot_radius
            h_minus = obs.get_signed_distance(p_minus) - self.robot_radius
            grad_h[i] = (h_plus - h_minus) / (2 * self.epsilon)
        
        # First Lie Derivative: LfH = grad_h^T * f_pos
        # where f_pos = [v*cos(theta), v*sin(theta)]^T
        f_pos = np.array([v * np.cos(theta), v * np.sin(theta)])
        LfH = grad_h.T @ f_pos
        
        # Second Lie Derivative terms
        # LgLfH = [d(LfH)/dv, d(LfH)/dtheta]
        # d(LfH)/dv = grad_h^T * [cos(theta), sin(theta)]
        dLfH_dv = grad_h.T @ np.array([np.cos(theta), np.sin(theta)])
        
        # d(LfH)/dtheta = grad_h^T * [-v*sin(theta), v*cos(theta)]
        dLfH_dtheta = grad_h.T @ np.array([-v * np.sin(theta), v * np.cos(theta)])
        
        LgLfH = np.array([dLfH_dv, dLfH_dtheta])  # 1x2 array
        
        # Lf^2H: second Lie derivative (control-independent)
        # Lf^2H = d(LfH)/dx * f_x + d(LfH)/dy * f_y + d(LfH)/dtheta * f_theta + d(LfH)/dv * f_v
        # Since f = [v*cos(theta), v*sin(theta), 0, 0], we have:
        # Lf^2H = d(LfH)/dx * v*cos(theta) + d(LfH)/dy * v*sin(theta)
        
        # For computational efficiency, we use a simplified approximation:
        # Lf^2H ≈ 0 for low velocities, or we can compute it more efficiently
        # Here we compute it using the fact that:
        # LfH = grad_h^T * [v*cos(theta), v*sin(theta)]
        # d(LfH)/dx = d(grad_h)/dx^T * f_pos + grad_h^T * d(f_pos)/dx
        # Since f_pos doesn't depend on x, d(f_pos)/dx = 0
        # So d(LfH)/dx = d(grad_h)/dx^T * f_pos
        
        # Compute Hessian of h (second derivatives) for more accurate Lf^2H
        # For efficiency, we approximate Lf^2H ≈ 0 when v is small
        # This is a common approximation in practice
        if v < 0.1:
            Lf2H = 0.0  # Approximation for low velocities
        else:
            # Compute second derivatives of h with respect to position
            hess_h = np.zeros((2, 2))
            for i in range(2):
                for j in range(2):
                    # Second partial derivative: d^2h/(dx_i dx_j)
                    x_ij_pp = x.copy()
                    x_ij_pp[i] += self.epsilon
                    x_ij_pp[j] += self.epsilon
                    x_ij_pm = x.copy()
                    x_ij_pm[i] += self.epsilon
                    x_ij_pm[j] -= self.epsilon
                    x_ij_mp = x.copy()
                    x_ij_mp[i] -= self.epsilon
                    x_ij_mp[j] += self.epsilon
                    x_ij_mm = x.copy()
                    x_ij_mm[i] -= self.epsilon
                    x_ij_mm[j] -= self.epsilon
                    
                    h_pp = obs.get_signed_distance(x_ij_pp[:2]) - self.robot_radius
                    h_pm = obs.get_signed_distance(x_ij_pm[:2]) - self.robot_radius
                    h_mp = obs.get_signed_distance(x_ij_mp[:2]) - self.robot_radius
                    h_mm = obs.get_signed_distance(x_ij_mm[:2]) - self.robot_radius
                    
                    hess_h[i, j] = (h_pp - h_pm - h_mp + h_mm) / (4 * self.epsilon ** 2)
            
            # Lf^2H = f_pos^T * hess_h * f_pos + grad_h^T * d(f_pos)/dx * f_pos
            # Since d(f_pos)/dx = 0, we have:
            Lf2H = f_pos.T @ hess_h @ f_pos
        
        return h, LfH, LgLfH, Lf2H

    def _get_cbf_qp_constraints(self, x: np.ndarray) -> List[Tuple[np.ndarray, float]]:
        """
        Calculates CBF QP constraints for all obstacles.
        
        For relative degree 2 CBF, the constraint is:
        LgLfH * u >= -Lf^2H - (gamma + alpha) * LfH - gamma * alpha * h
        
        Returns:
            List of (A_cbf, b_cbf) tuples where A_cbf @ u >= b_cbf
        """
        constraints = []
        
        for obs in self.env.obstacles:
            h, LfH, LgLfH, Lf2H = self._get_cbf_derivatives(x, obs)
            
            # Only check active constraints (h is near zero or negative)
            # This reduces computational burden
            if h <= 1.0:  # Check obstacles within 1.0m
                # CBF constraint: LgLfH * u >= -Lf^2H - (gamma + alpha) * LfH - gamma * alpha * h
                b_cbf = -Lf2H - (self.gamma_cbf + self.alpha_cbf) * LfH - self.gamma_cbf * self.alpha_cbf * h
                
                constraints.append((LgLfH.reshape(1, 2), b_cbf))
                
        return constraints
    
    def is_clf_cbf_compatible(self, x: np.ndarray, q: np.ndarray, 
                             check_segment: bool = True, 
                             num_checks: int = 5) -> bool:
        """
        Checks feasibility of the combined CLF-CBF QP.
        
        Args:
            x: Current state [x, y, theta, v]
            q: Goal state (can be 2D [x, y] or 4D)
            check_segment: If True, check multiple points along the path
            num_checks: Number of points to check along segment
        
        Returns:
            True if QP is feasible, False otherwise
        """
        if check_segment:
            # Check multiple points along the segment from current state
            # For simplicity, we'll check at the current state and interpolate toward goal
            states_to_check = [x]
            
            # If we have a parent state, check intermediate points
            # For now, just check the current state
            for state in states_to_check:
                if not self._solve_qp_at_state(state, q):
                    return False
            return True
        else:
            return self._solve_qp_at_state(x, q)
    
    def _solve_qp_at_state(self, x: np.ndarray, q: np.ndarray) -> bool:
        """Solves the QP at a single state."""
        try:
            # 1. Define optimization variable
            u = cp.Variable(2)  # u = [a, omega]
            
            # 2. Get constraints
            constraints = []
            slack_terms = []
            
            # Check distance to goal - if far, make CLF constraint optional/lenient
            if len(q) == 2:
                q_pos = q
            else:
                q_pos = q[:2]
            pos_dist = np.linalg.norm(x[:2] - q_pos)
            use_clf = pos_dist < 5.0  # Only enforce CLF when within 5m of goal
            
            # 3.1. CLF Constraint: A_clf * u <= b_clf
            if use_clf:
                A_clf, b_clf = self._get_clf_derivatives(x, q)
                
                # Add slack to CLF constraint for robustness (allow slight violation)
                slack_clf = cp.Variable(1, nonneg=True)
                constraints.append(A_clf @ u <= b_clf + slack_clf)
                slack_terms.append(100.0 * cp.sum_squares(slack_clf))
            
            # 3.2. CBF Constraints: A_cbf * u >= b_cbf
            cbf_constraints = self._get_cbf_qp_constraints(x)
            for A_cbf, b_cbf in cbf_constraints:
                # Add slack to CBF constraints (but penalize heavily to maintain safety)
                slack_cbf = cp.Variable(1, nonneg=True)
                constraints.append(A_cbf @ u >= b_cbf - slack_cbf)
                slack_terms.append(10000.0 * cp.sum_squares(slack_cbf))  # Heavy penalty for safety violations
                
            # 3.3. Input Constraints
            constraints.append(u[0] >= -self.a_max)  # linear accel
            constraints.append(u[0] <= self.a_max)
            constraints.append(u[1] >= -self.omega_max)  # angular vel
            constraints.append(u[1] <= self.omega_max)
            
            # 2. Objective (Minimize control effort + slack penalties)
            if slack_terms:
                objective = cp.Minimize(cp.sum_squares(u) + cp.sum(slack_terms))
            else:
                objective = cp.Minimize(cp.sum_squares(u))
            
            # 4. Solve the QP
            prob = cp.Problem(objective, constraints)
            prob.solve(verbose=False, solver=cp.OSQP)
            
            # Feasibility check
            if prob.status in ["optimal", "optimal_inaccurate"]:
                return True
            else:
                return False
                
        except Exception:
            # If solver fails, assume infeasible
            return False


class CCLFCBFRRT:
    """
    C-CLF-CBF-RRT Planner for unicycle model.
    
    Builds a tree of states that are both CLF-CBF compatible and collision-free.
    """
    
    def __init__(self, env: WarehouseEnv, config: Dict[str, Any], 
                 max_iter: int = 5000, 
                 step_size: float = 0.5,
                 goal_tolerance: float = 0.5):
        """
        Initialize the planner.
        
        Args:
            env: Warehouse environment
            config: Configuration dictionary
            max_iter: Maximum RRT iterations
            step_size: Maximum step size for steering (eta)
            goal_tolerance: Distance threshold for goal reaching
        """
        self.env = env
        self.config = config
        self.max_iter = max_iter
        self.step_size = step_size
        self.goal_tolerance = goal_tolerance
        
        self.core = UnicyclePlannerCore(env, config)
        self.tree: List[RRTNode] = []
        
    def _distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute weighted distance between two states."""
        # Use position and orientation for distance
        pos_diff = np.linalg.norm(x1[:2] - x2[:2])
        theta_diff = min(abs(x1[2] - x2[2]), 2 * np.pi - abs(x1[2] - x2[2]))
        v_diff = abs(x1[3] - x2[3])
        
        # Weighted combination
        return pos_diff + 0.1 * theta_diff + 0.1 * v_diff
    
    def _nearest_neighbor(self, x_rand: np.ndarray) -> RRTNode:
        """Find nearest node in tree to x_rand."""
        if len(self.tree) == 0:
            raise ValueError("Tree is empty")
        
        min_dist = float('inf')
        nearest = None
        
        for node in self.tree:
            dist = self._distance(node.state, x_rand)
            if dist < min_dist:
                min_dist = dist
                nearest = node
        
        return nearest
    
    def _steer(self, x_near: np.ndarray, x_rand: np.ndarray) -> np.ndarray:
        """
        Steer from x_near toward x_rand with maximum step size.
        
        Returns:
            New state x_new
        """
        dist = self._distance(x_near, x_rand)
        
        if dist <= self.step_size:
            return x_rand.copy()
        else:
            # Interpolate
            alpha = self.step_size / dist
            x_new = x_near + alpha * (x_rand - x_near)
            
            # Ensure velocity is non-negative
            x_new[3] = max(0.0, x_new[3])
            
            # Normalize theta to [0, 2*pi)
            x_new[2] = x_new[2] % (2 * np.pi)
            
            return x_new
    
    def _collision_free(self, x1: np.ndarray, x2: np.ndarray, num_checks: int = 10) -> bool:
        """
        Check if path between x1 and x2 is collision-free.
        
        Args:
            x1: Start state
            x2: End state
            num_checks: Number of intermediate points to check
        
        Returns:
            True if path is collision-free
        """
        for i in range(num_checks + 1):
            alpha = i / num_checks
            x_interp = x1 + alpha * (x2 - x1)
            x_interp[2] = x_interp[2] % (2 * np.pi)
            x_interp[3] = max(0.0, x_interp[3])
            
            # Convert to NN state for collision checking
            nn_state = self.env.get_nn_state(x_interp)
            h_star = self.env.get_ground_truth_safety(nn_state)
            
            if h_star <= 0:
                return False
        
        return True
    
    def plan(self, x_init: np.ndarray, x_goal: np.ndarray) -> Optional[List[np.ndarray]]:
        """
        Plan a path from x_init to x_goal.
        
        Args:
            x_init: Initial state [x, y, theta, v]
            x_goal: Goal state (can be 2D [x, y] or 4D)
        
        Returns:
            List of states forming the path, or None if planning fails
        """
        # Ensure x_init is 4D
        if len(x_init) == 2:
            x_init = np.array([x_init[0], x_init[1], 0.0, 0.0])
        elif len(x_init) != 4:
            raise ValueError(f"x_init must be 2D or 4D, got {len(x_init)}D")
        
        # Initialize tree with root node
        root = RRTNode(state=x_init.copy(), parent=None, cost=0.0)
        self.tree = [root]
        
        for _ in range(self.max_iter):
            # 1. Sample random state
            x_rand = self.env.sample_random_state()
            print(f"Sampled random state: {x_rand}")
            # 2. Find nearest neighbor
            x_near_node = self._nearest_neighbor(x_rand)
            x_near = x_near_node.state
            print(f"Nearest neighbor: {x_near}")
            # 3. Steer
            x_new = self._steer(x_near, x_rand)
            print(f"Steered: {x_new}")
            
            # 4. Collision check
            if not self._collision_free(x_near, x_new):
                print("  ✗ Collision check failed")
                continue
            
            print("  ✓ Collision check passed")
            
            # 5. CLF-CBF Compatibility Check
            is_compatible = self.core.is_clf_cbf_compatible(x_new, x_goal, check_segment=False)
            if not is_compatible:
                print("  ✗ CLF-CBF compatibility check failed")
                continue
            
            print("  ✓ CLF-CBF compatibility check passed")
            
            # 6. Add node to tree
            cost_new = x_near_node.cost + self._distance(x_near, x_new)
            new_node = RRTNode(state=x_new.copy(), parent=x_near_node, cost=cost_new)
            self.tree.append(new_node)
            
            # 7. Goal check
            goal_pos = x_goal[:2] if len(x_goal) >= 2 else x_goal
            new_pos = x_new[:2]
            if np.linalg.norm(new_pos - goal_pos) < self.goal_tolerance:
                # Reconstruct path
                return self._reconstruct_path(new_node)
        
        # Planning failed
        return None
    
    def _reconstruct_path(self, node: RRTNode) -> List[np.ndarray]:
        """Reconstruct path from root to given node."""
        path = []
        current = node
        
        while current is not None:
            path.append(current.state.copy())
            current = current.parent
        
        path.reverse()
        return path
    
    def get_tree(self) -> List[RRTNode]:
        """Get the current RRT tree."""
        return self.tree

