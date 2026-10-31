"""실제 two-body 실행을 기록해 함수 사이 데이터 흐름도(HTML)를 만들어요.

    python tools/flow_map/build_flow_map.py             # 하중 4단계
    python tools/flow_map/build_flow_map.py --steps 20

결과는 tools/flow_map/flow_map.html이고 브라우저로 열면 돼요.
src 코드는 바꾸지 않고, 실행하는 동안만 몇몇 함수를 감싸서 주고받는 값을 기록해요.
수치와 코드 줄 번호는 실행할 때마다 새로 채워지지만, 노드 배치·설명·수식은
template.html에서 직접 관리해요. 함수 이름이나 호출 흐름이 바뀌면 두 파일을 같이 고쳐 주세요.
structured_bfgs_gap은 명령줄 --methods에 없어서 compression.run()을 직접 불러 기록해요.
충돌 탐색에 ipctk가 필요해요.
"""
import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import src.collision.ipc_toolkit as ipc_toolkit  # noqa: E402
import src.geometry.contact_patches as contact_patches  # noqa: E402
import src.simulation.compression as compression  # noqa: E402
import src.simulation.two_body as two_body  # noqa: E402
from src.geometry.surface import bottom_faces  # noqa: E402

METHODS = ('full', 'gn', 'filtered_gn', 'sr1_gap', 'structured_bfgs_gap')
SKIN = .1
# template.html의 {{경로#찾을 글자}}를 그 글자가 처음 나오는 줄 번호로 바꿔요.
CODE_REF = re.compile(r'\{\{((?:src/|run\.py)[^#}]*)(?:#([^}]+))?\}\}')


def wrap(owner, name, make):
    """owner.name을 기록용 함수로 감싸고, 이름이 사라졌으면 고칠 곳을 알려줘요."""
    original = getattr(owner, name, None)
    if not callable(original):
        raise SystemExit(f'{owner.__name__}.{name}을 찾지 못했어요. 코드에서 이름이 바뀌었다면 '
                         'build_flow_map.py의 기록 지점과 template.html을 같이 고쳐 주세요.')
    setattr(owner, name, make(original))


def record(steps):
    """네 헤시안 방식으로 압축 시뮬레이션을 돌리며 호출 기록과 메시 정보를 모아요."""
    calls, meta, current = [], {}, {}

    def log(kind, **data):
        calls.append(dict(kind=kind, method=current.get('method'), **data))

    def on_refresh(original):
        def refresh(self, positions, before_change=None):
            info = original(self, positions, before_change)
            meta.setdefault('groups', len(self.groups))
            log('refresh', **info)
            return info
        return refresh

    def on_skin_cap(original):
        def max_step(self, start, end):
            value = original(self, start, end)
            movement = float(np.linalg.norm(np.asarray(end) - np.asarray(start), axis=1).max())
            log('skin_cap', value=float(value), movement=movement)
            return value
        return max_step

    def on_ccd(original):
        def max_step(self, start, end):
            value = original(self, start, end)
            meta.setdefault('ipctk', str(self.info.get('version', '?')))
            log('ccd', value=float(value))
            return value
        return max_step

    def on_contact(original):
        def contact_terms(*args, **kwargs):
            result = original(*args, **kwargs)
            if kwargs.get('compute_hessian', True):
                gaps = np.asarray(result['gaps'])
                log('contact', active=int(np.sum(gaps < kwargs.get('d_hat', .2))),
                    min_gap=float(gaps.min()) if len(gaps) else None,
                    energy=float(result['energy']), hess_nnz=int(result['hessian'].nnz),
                    contact_dofs=int(result['contact_dofs']), filter_count=int(result['filter_count']))
            return result
        return contact_terms

    def on_solve(original):
        def solve(problem, method='gn', initial=None, **kwargs):
            objective = problem['objective']

            def marked(state, name, model=None):
                # 반복마다 한 번 불리므로 앞뒤 기록을 반복 단위로 나누는 표시로 써요.
                log('objective')
                return objective(state, name, model)
            problem['objective'] = marked
            current['method'] = method
            log('solve_start', closure=float(problem['closure']))
            try:
                result = original(problem, method, initial=initial, **kwargs)
            finally:
                problem['objective'] = objective
            forces = problem['diagnostics'](result['state'])
            log('solve_end', status=result['status'], updates=result['updates'], trace=result['trace'],
                Fz_upper=float(forces['force_upper'][2]), Fz_lower=float(forces['force_lower'][2]))
            return result
        return solve

    wrap(contact_patches.AutomaticRIMLSContact, 'refresh', on_refresh)
    wrap(contact_patches.AutomaticRIMLSContact, 'max_step', on_skin_cap)
    wrap(ipc_toolkit.IPCMeshCollision, 'max_step', on_ccd)
    wrap(two_body, 'multiple_contact_terms_sparse', on_contact)
    wrap(compression, 'solve', on_solve)
    problem, _, _ = compression.run(steps=steps, methods=METHODS, skin=SKIN)

    lower, upper = problem['meshes']
    contact_tris = [len(lower['top_faces']), len(bottom_faces(upper))]
    meta.update(steps=steps, closure=float(problem['closure']), initial_gap=float(problem['initial_gap']),
                d_hat=float(problem['d_hat']), skin=SKIN, mesh_counts=[int(n) for n in problem['mesh_counts']],
                nodes=len(problem['reference'][0]), tets=len(lower['tetrahedra']),
                bfaces=len(lower['boundary_faces']), contact_tris=contact_tris, samples=3 * sum(contact_tris),
                dofs=len(problem['initial']), total_dofs=len(problem['reference_flat']),
                nnz=int(problem['stiffness'].nnz), nnz_free=int(problem['stiffness_free'].nnz))
    return calls, meta


def rounded(value, digits):
    """유효숫자를 줄이고, JSON에 담을 수 없는 값은 None으로 바꿔요."""
    if value is None or not np.isfinite(value):
        return None
    return float('%.*g' % (digits, value))


def summarize(calls):
    """호출 기록을 방식 → 하중 단계 → Newton 반복 순서로 묶어요."""
    methods = {name: [] for name in METHODS}
    solve = None
    for call in calls:
        kind = call['kind']
        if kind == 'solve_start':
            solve = dict(closure=call['closure'], refresh={}, iterations=[])
        elif solve is None:
            continue
        elif kind == 'refresh':
            solve['refresh'] = call
        elif kind == 'objective':
            solve['iterations'].append(dict(refresh=solve['refresh']))
        elif kind in ('contact', 'ccd', 'skin_cap') and solve['iterations']:
            solve['iterations'][-1][kind] = call
        elif kind == 'solve_end':
            methods[call['method']].append(load_step(solve, call))
            solve = None
    return methods


def load_step(solve, end):
    """한 하중 단계의 Newton 반복마다 템플릿이 쓰는 값만 남겨요."""
    iterations = []
    for trace, seen in zip(end['trace'], solve['iterations']):
        patch, contact = seen['refresh'], seen.get('contact')
        item = dict(r=rounded(trace['residual'], 3), E=rounded(trace['energy'], 5),
                    cand=patch.get('candidate_count', 0), pairs=patch.get('candidate_face_pairs', 0),
                    q=patch.get('contact_samples', 0), chg=bool(patch.get('patch_changed', False)),
                    Ec=0., gap=None, act=0, nnz=0, cd=0, fc=0)
        if contact:
            item.update(Ec=rounded(contact['energy'], 4), gap=rounded(contact['min_gap'], 5), act=contact['active'],
                        nnz=contact['hess_nnz'], cd=contact['contact_dofs'], fc=contact['filter_count'])
        if 'alpha' in trace:
            ccd, cap = seen.get('ccd', {}), seen.get('skin_cap', {})
            item.update(a=rounded(trace['alpha'], 4), bt=int(trace['backtracks']),
                        reg=rounded(trace.get('regularization', 0.), 3), ccd=rounded(ccd.get('value', 1.), 4),
                        skin=rounded(cap.get('value', 1.), 4), mv=rounded(cap.get('movement', 0.), 4))
        if 'sr1_factor_count' in trace:
            item.update(sf=int(trace['sr1_factor_count']), sn=rounded(trace.get('sr1_correction_norm', 0.), 3))
        if 'bfgs_correction_norm' in trace:
            item.update(bu=int(trace['bfgs_updates']), br=int(trace['bfgs_resets']),
                        bn=rounded(trace['bfgs_correction_norm'], 3))
        iterations.append(item)
    return dict(delta=round(solve['closure'], 6), st=end['status'], upd=end['updates'],
                Fzu=rounded(end['Fz_upper'], 5), Fzl=rounded(end['Fz_lower'], 5), its=iterations)


def revision():
    """현재 커밋과 커밋 안 된 수정이 있는지 알려줘요."""
    try:
        def git(*args):
            return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        return git('rev-parse', '--short', 'HEAD'), bool(git('status', '--porcelain', '--untracked-files=no'))
    except (OSError, subprocess.CalledProcessError):
        return '—', False


def resolve_code_refs(text):
    """{{경로#찾을 글자}}를 '경로:줄'로 바꾸고, 못 찾은 위치는 모아서 알려줘요."""
    files, missing = {}, []

    def replace(match):
        path, anchor = match.group(1), match.group(2)
        if anchor is None:
            return path
        if path not in files:
            try:
                files[path] = (ROOT / path).read_text(encoding='utf-8').splitlines()
            except OSError:
                files[path] = []
        for number, line in enumerate(files[path], 1):
            if anchor in line:
                return f'{path}:{number}'
        missing.append(f'{path} 안의 "{anchor}"')
        return path

    text = CODE_REF.sub(replace, text)
    if missing:
        raise SystemExit('template.html이 가리키는 코드 위치를 찾지 못했어요:\n  ' + '\n  '.join(missing))
    return text


def render(meta, methods, out):
    template = (HERE / 'template.html').read_text(encoding='utf-8')
    if template.count('__DATA__') != 1:
        raise SystemExit('template.html에는 __DATA__ 자리가 정확히 하나 있어야 해요.')
    data = json.dumps(dict(meta=meta, methods=methods), ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    out.write_text(resolve_code_refs(template).replace('__DATA__', data.replace('</', '<\\/')), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--steps', type=int, default=4, help='하중 단계 수 (run.py two-body의 --steps, 기본 4)')
    parser.add_argument('--out', type=Path, default=HERE / 'flow_map.html', help='만들 HTML 경로')
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps는 1 이상이어야 해요')
    calls, meta = record(args.steps)
    rev, dirty = revision()
    meta.update(rev=rev, dirty=dirty, recorded=datetime.now().strftime('%Y-%m-%d %H:%M'),
                cmd=f'python tools/flow_map/build_flow_map.py --steps {args.steps}')
    methods = summarize(calls)
    render(meta, methods, args.out)
    counts = ', '.join(f'{name} {sum(len(step["its"]) for step in steps)}회' for name, steps in methods.items())
    print(f'흐름도를 만들었어요: {args.out}\n  Newton 반복 수: {counts}')


if __name__ == '__main__':
    main()
