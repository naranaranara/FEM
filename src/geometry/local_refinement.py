"""##r## Conforming local edge bisection on the existing polyhedral geometry.
Every tetrahedron sharing a split edge is bisected: no hanging nodes.
New surface nodes stay on existing flat faces, not the analytic cylinder.
"""
import numpy as np

_EDGES = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])


def _split_faces(faces, split_edges, midpoint_ids, node_count):
    """Split tagged faces with the same midpoint IDs as the volume mesh."""
    faces = np.asarray(faces, dtype=int)
    pairs = np.array([[0, 1], [1, 2], [2, 0]])
    keys = split_edges[:, 0]*node_count+split_edges[:, 1]
    edges = np.sort(faces[:, pairs], axis=2)
    face_keys = edges[:, :, 0]*node_count+edges[:, :, 1]
    at = np.searchsorted(keys, face_keys)
    hit = (at < len(keys)) & (keys[np.minimum(at, len(keys)-1)] == face_keys)
    if np.any(hit.sum(axis=1) > 1):
        raise RuntimeError('simultaneous bisections must not share a face')
    rows = np.flatnonzero(hit.any(axis=1))
    if not len(rows):
        return faces.copy()
    choices = np.argmax(hit[rows], axis=1)
    pair = pairs[choices]
    mid = midpoint_ids[at[rows, choices]]
    a, b = faces[rows].copy(), faces[rows].copy()
    a[np.arange(len(rows)), pair[:, 0]] = mid
    b[np.arange(len(rows)), pair[:, 1]] = mid
    return np.concatenate((faces[~hit.any(axis=1)], a, b))


def refine_cylinder_contact(mesh, target_size, half_width=.4, depth=.3,
                            max_rounds=100, max_nodes=500000, progress=None):
    """##r## Limit edge length in cells intersecting a box at the contact pole.

    The box uses local x/y and depth from the facing pole. Incident neighbour
    cells are also split for conformity. Keep original nodes and boundary tags.
    """
    if any(not np.isfinite(v) or v <= 0 for v in (target_size, half_width, depth)):
        raise ValueError('positive finite target_size, half_width and depth required')
    if half_width >= min(mesh['radius'], mesh['half_length']) or depth >= mesh['radius']:
        raise ValueError('refinement box must stay near the facing pole')
    if not isinstance(max_rounds, int) or max_rounds < 1:
        raise ValueError('positive max_rounds required')
    result = dict(mesh)
    nodes, local, tets = (mesh[k].copy() for k in ('nodes', 'local_nodes', 'tetrahedra'))
    tags = {k: mesh[k].copy() for k in ('boundary_faces', 'side_faces', 'end_faces',
            'contact_faces', 'query_faces', 'top_faces')}
    groups = {k: mesh[k].copy() for k in ('support_nodes', 'bottom_nodes')}
    pole = local[:, 2].max() if mesh['facing'] == 'up' else local[:, 2].min()
    sign = -1. if mesh['facing'] == 'up' else 1.
    original_nodes, original_tets = len(nodes), len(tets)
    for iteration in range(max_rounds+1):
        box_coords = local.copy()
        box_coords[:, 2] = sign*(local[:, 2]-pole)
        xyz = box_coords[tets]
        lo, hi = xyz.min(axis=1), xyz.max(axis=1)
        selected = ((lo[:, 0] < half_width) & (hi[:, 0] > -half_width) &
                    (lo[:, 1] < half_width) & (hi[:, 1] > -half_width) &
                    (lo[:, 2] < depth) & (hi[:, 2] >= 0.))
        edges, inverse, counts = np.unique(np.sort(tets[:, _EDGES], axis=2).reshape(-1, 2),
                                           axis=0, return_inverse=True, return_counts=True)
        inverse = inverse.reshape(-1, 6)
        lengths = np.linalg.norm(nodes[edges[:, 1]]-nodes[edges[:, 0]], axis=1)
        local_max = float(lengths[inverse[selected]].max()) if selected.any() else 0.
        candidate = np.zeros(len(edges), dtype=bool)
        candidate[inverse[selected].ravel()] = True
        candidate &= lengths > target_size*(1+1e-12)
        if not candidate.any():
            break
        if iteration == max_rounds:
            raise RuntimeError('local refinement did not reach target_size')
        priority = np.zeros(len(edges), dtype=int)
        ids = np.flatnonzero(candidate)
        ids = ids[np.lexsort((ids, lengths[ids]))]
        priority[ids] = np.arange(1, len(ids)+1)
        choices = np.argmax(priority[inverse], axis=1)
        winner = inverse[np.arange(len(tets)), choices]
        votes = np.bincount(winner[priority[winner] > 0], minlength=len(edges))
        accepted = candidate & (votes == counts)
        chosen = np.flatnonzero(accepted)
        if not len(chosen):
            raise RuntimeError('no compatible refinement edges')
        old_n = len(nodes)
        if old_n+len(chosen) > max_nodes:
            raise ValueError('local refinement exceeds max_nodes')
        split_edges = edges[chosen]
        mids = np.arange(old_n, old_n+len(chosen))
        for name, faces in tags.items():
            tags[name] = _split_faces(faces, split_edges, mids, old_n)
        for name, group in groups.items():
            on_group = np.isin(split_edges, group).all(axis=1)
            groups[name] = np.r_[group, mids[on_group]]
        mapping = np.full(len(edges), -1, dtype=int)
        mapping[chosen] = mids
        rows = np.flatnonzero(accepted[winner])
        pair = _EDGES[choices[rows]]
        a, b = tets[rows].copy(), tets[rows].copy()
        a[np.arange(len(rows)), pair[:, 0]] = mapping[winner[rows]]
        b[np.arange(len(rows)), pair[:, 1]] = mapping[winner[rows]]
        keep = np.ones(len(tets), dtype=bool)
        keep[rows] = False
        tets = np.concatenate((tets[keep], a, b))
        nodes = np.vstack((nodes, nodes[split_edges].mean(axis=1)))
        local = np.vstack((local, local[split_edges].mean(axis=1)))
        if progress is not None and iteration % 5 == 0:
            progress('local refinement round=%d nodes=%d cells=%d' %
                     (iteration+1, len(nodes), len(tets)))
    result.update(nodes=nodes, local_nodes=local, tetrahedra=tets, **tags, **groups)
    result['local_refinement'] = dict(target_size=target_size, half_width=half_width,
        depth=depth, rounds=iteration, max_local_edge=local_max,
        original_nodes=original_nodes, original_tetrahedra=original_tets,
        nodes=len(nodes), tetrahedra=len(tets), geometry='original piecewise planar surface')
    return result
