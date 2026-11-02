"""##r## Prescribed Hertz pressure on the EXISTING finite-cylinder FEM mesh.

No RIMLS, barrier, nonlinear contact solve, or fitted load normalization.
Run: python -B -m src.simulation.test_cylinder_compliance
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np
from src.mechanics.sparse_factor import factorize_sparse  # ##r##
from threadpoolctl import threadpool_limits
from src.geometry.mesh import make_cylinder
from src.geometry.primitives import crossed_cylinder_curvature
from src.mechanics.hertz import elliptical_hertz
from src.mechanics.solid import assemble_stiffness


def hertz_quadrature(reference, directions, order):
    """##r## Integrate pressure over the projected ellipse, not curved area.

    t=sqrt(1-rho**2) gives p*dA=p0*a*b*t**2*dt*dtheta.
    Thus total load is integrated exactly (up to roundoff) for order >= 2.
    Refinement is still needed for the piecewise-linear nodal shape functions.
    """
    if order < 2:
        raise ValueError('quadrature order must be >= 2')
    x, w = np.polynomial.legendre.leggauss(order)
    t, w = (x+1)/2, w/2
    theta = (np.arange(4*order)+.5)*2*np.pi/(4*order)
    rho = np.sqrt(1-t*t)
    a, b = reference['major_radius'], reference['minor_radius']
    local = np.stack(np.broadcast_arrays(a*rho[:, None]*np.cos(theta),
                                        b*rho[:, None]*np.sin(theta)), axis=-1)
    xy = local.reshape(-1, 2) @ directions.T
    weights = np.broadcast_to((reference['peak_pressure']*a*b*t*t*w)[:, None],
                              (order, 4*order)).ravel()*2*np.pi/(4*order)
    return xy, weights


def pressure_load(nodes, faces, reference, directions, order, sign):
    """##r## Transfer projected pressure to original triangle nodes with N_i.

    Only facing-side faces are supplied. Every quadrature point must be covered.
    Points on shared edges are assigned once; no duplicate contact load.
    """
    xy, weights = hertz_quadrature(reference, directions, order)
    assigned = np.zeros(len(xy), dtype=bool)
    force = np.zeros_like(nodes)
    for face in faces:
        triangle = nodes[face, :2]
        ids = np.flatnonzero(~assigned & np.all(xy >= triangle.min(0)-1e-12, axis=1)
                            & np.all(xy <= triangle.max(0)+1e-12, axis=1))
        if not len(ids):
            continue
        matrix = (triangle[1:]-triangle[0]).T
        if abs(np.linalg.det(matrix)) < 1e-16:
            continue
        uv = np.linalg.solve(matrix, (xy[ids]-triangle[0]).T).T
        bary = np.column_stack((1-uv.sum(axis=1), uv))
        inside = np.all(bary >= -1e-10, axis=1)
        ids, bary = ids[inside], bary[inside]
        force[face, 2] += sign*(weights[ids, None]*bary).sum(axis=0)
        assigned[ids] = True
    if not np.all(assigned):
        raise ValueError('Hertz ellipse is not covered by facing surface: %d points' %
                         np.count_nonzero(~assigned))
    return force.ravel()


def solve_body(mesh, reference, directions, young, poisson, sign,
               orders=(16, 32, 64, 128), load_tol=.002, linear_backend='superlu'):
    """##r## Check nodal-load quadrature, then solve K_ff u_f=f_f once."""
    nodes = mesh['nodes']
    faces = mesh['side_faces']
    xyz = nodes[faces]
    nz = np.cross(xyz[:, 1]-xyz[:, 0], xyz[:, 2]-xyz[:, 0])[:, 2]
    faces = faces[nz*(-sign) > 0]
    previous = None
    quadrature = []
    for order in orders:
        force = pressure_load(nodes, faces, reference, directions, order, sign)
        change = None if previous is None else float(np.linalg.norm(force-previous, 1)/reference['force'])
        quadrature.append(dict(order=order, relative_nodal_load_change=change))
        if change is not None and change < load_tol:
            break
        previous = force
    else:
        raise ValueError('Nodal pressure quadrature has not converged: %s' % quadrature)
    print('  pressure quadrature:', quadrature, flush=True)
    K = assemble_stiffness(nodes, mesh['tetrahedra'], young=young, poisson=poisson, sparse=True)
    fixed = (3*mesh['support_nodes'][:, None]+np.arange(3)).ravel()
    free = np.setdiff1d(np.arange(nodes.size), fixed)
    u = np.zeros(nodes.size)
    # ##r## Reuse the existing PARDISO adapter; FEM and loading stay identical.
    tick = time.perf_counter()
    factor = factorize_sparse(K[free][:, free], backend=linear_backend)
    try:
        u[free] = factor.solve(force[free])
    finally:
        if hasattr(factor, 'close'):
            factor.close()
    linear_s = time.perf_counter()-tick
    residual = K @ u-force
    reaction = np.zeros_like(force)
    reaction[fixed] = residual[fixed]
    facing_nodes = np.unique(faces)
    center = int(facing_nodes[np.argmin(np.linalg.norm(nodes[facing_nodes, :2], axis=1))])
    if np.linalg.norm(nodes[center, :2]) > 1e-10:
        raise ValueError('Expected a surface node at the contact center')
    applied = force.reshape(-1, 3).sum(0)
    balance = (reaction+force).reshape(-1, 3).sum(0)
    moment = np.cross(nodes, (reaction+force).reshape(-1, 3)).sum(0)
    return dict(linear_backend=linear_backend, linear_s=linear_s,
                nodes=len(nodes), tetrahedra=len(mesh['tetrahedra']), free_dofs=len(free),
                center_node=center, center_displacement=u.reshape(-1, 3)[center].tolist(),
                applied_force=applied.tolist(), reaction=reaction.reshape(-1, 3).sum(0).tolist(),
                force_relative_error=float(abs(abs(applied[2])/reference['force']-1)),
                relative_free_residual=float(np.linalg.norm(residual[free])/np.linalg.norm(force[free])),
                force_balance=balance.tolist(), moment_balance=moment.tolist(),
                elastic_energy=float(.5*u @ (K @ u)), external_work=float(u @ force),
                quadrature=quadrature), u.reshape(-1, 3)


def run(n_angle=128, n_axis=25, n_radial=1, delta=.005, outdir=None, radial_grading=1.,
        local_size=None, local_half_width=.4, local_depth=.3, linear_backend='pardiso'):
    """##r## Same geometry/material/end clamps as the current cylinder study.

    Grips are fixed at zero here; known separating pressure is applied instead
    of imposed grip closure. This isolates finite-body compliance. Hertz's
    half-space indentation is a reference, not an exact finite-cylinder solution.
    """
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError('positive finite delta required')
    if linear_backend not in ('superlu', 'pardiso'):
        raise ValueError('linear backend must be superlu or pardiso')
    _, curvature, directions = crossed_cylinder_curvature(3., 5., 45.)
    reference = elliptical_hertz(delta, curvature, young_lower=30., young_upper=30., poisson=.3)
    parameters = dict(n_angle=n_angle, n_axis=n_axis, n_radial=n_radial, radial_grading=radial_grading, delta=delta,
                      radii=[3., 5.], half_length=3., angle=45., young=[30., 30.], poisson=.3,
                      local_size=local_size, local_half_width=local_half_width,
                      local_depth=local_depth, linear_backend=linear_backend)
    bodies, arrays = [], {}
    with threadpool_limits(1):
        for name, radius, facing, angle, clearance, sign in (
                ('lower', 3., 'up', 0., 0., -1.), ('upper', 5., 'down', 45., .03, 1.)):
            print('Solving %s cylinder, n_radial=%d, radial_grading=%g' %
                  (name, n_radial, radial_grading), flush=True)
            mesh = make_cylinder(radius=radius, half_length=3., n_angle=n_angle,
                                 n_axis=n_axis, n_radial=n_radial, radial_grading=radial_grading, facing=facing,
                                 angle_degrees=angle, clearance=clearance)
            # ##r## Refine only the contact neighbourhood, preserving geometry.
            if local_size is not None:
                from src.geometry.local_refinement import refine_cylinder_contact
                mesh = refine_cylinder_contact(mesh, local_size, local_half_width, local_depth,
                    progress=lambda msg: print('  '+msg, flush=True))
                print('  refined nodes=%d tetrahedra=%d' %
                      (len(mesh['nodes']), len(mesh['tetrahedra'])), flush=True)
            result, u = solve_body(mesh, reference, directions, 30., .3, sign,
                                   linear_backend=linear_backend)
            result['local_refinement'] = mesh.get('local_refinement')
            result['body'] = name
            bodies.append(result)
            arrays[name+'_nodes'] = mesh['nodes']
            arrays[name+'_displacement'] = u
            print('  center uz=%.9g, load error=%.3e, residual=%.3e' %
                  (result['center_displacement'][2], result['force_relative_error'],
                   result['relative_free_residual']), flush=True)
    measured = bodies[1]['center_displacement'][2]-bodies[0]['center_displacement'][2]
    result = dict(parameters=parameters, hertz=reference, bodies=bodies,
                  relative_center_displacement=measured, displacement_ratio=measured/delta,
                  interpretation='Prescribed-pressure finite-cylinder compliance diagnostic; '
                  'not a coupled contact validation or a half-space convergence proof.')
    print('Hertz delta=%.9g; FEM relative displacement=%.9g; ratio=%.6g' %
          (delta, measured, measured/delta), flush=True)
    if outdir is not None:
        folder = Path(outdir)
        folder.mkdir(parents=True, exist_ok=True)
        (folder/'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        np.savez_compressed(str(folder/'displacements.npz'), **arrays)
        print('Saved:', folder.resolve(), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-angle', type=int, default=128)
    parser.add_argument('--n-axis', type=int, default=25)
    parser.add_argument('--n-radial', type=int, default=1)
    parser.add_argument('--radial-grading', type=float, default=1.,
                        help='1: uniform; >1: concentrate radial layers near the surface (max 6)')
    parser.add_argument('--delta', type=float, default=.005)
    parser.add_argument('--outdir', default='experiments/cylinder_compliance_results/'+datetime.now().strftime('%Y-%m-%d_%H%M%S'))
    # ##r## Optional conforming local refinement; omitted => original mesh.
    parser.add_argument('--local-size', type=float, default=None,
                        help='maximum tetrahedron edge length near the contact pole')
    parser.add_argument('--local-half-width', type=float, default=.4)
    parser.add_argument('--local-depth', type=float, default=.3)
    parser.add_argument('--linear-backend', choices=('pardiso','superlu'), default='pardiso')
    run(**vars(parser.parse_args()))
