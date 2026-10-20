"""Guarded, backed-up patch of HKTex's eigen-alignment SVD call only."""
import argparse
import ast
from datetime import datetime
import difflib
import os
from pathlib import Path
import shutil
import tempfile

OLD = '        U, _, Vt = scipy.linalg.svd(correlation_matrix)'
NEW = '        U, _, Vt = _hktex_checked_alignment_svd(correlation_matrix)'
HELPER = Path(__file__).with_name('svd_helper.py.txt').read_text()


def patch(text):
    if NEW in text and HELPER.strip() in text:
        return text
    if '_hktex_checked_alignment_svd' in text or text.count(OLD) != 1:
        raise ValueError('Source differs from expected version; refusing to change it')
    fn = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'align_eigen')
    if OLD.strip() not in ast.get_source_segment(text, fn):
        raise ValueError('SVD call is not inside align_eigen')
    result = text.replace(OLD, NEW, 1).rstrip() + HELPER + '\n'
    compile(result, 'hktex/utils/lbo.py', 'exec')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', default='/root/hktex-work/hktex')
    p.add_argument('--apply', action='store_true')
    a = p.parse_args()
    target = Path(a.repo).resolve() / 'hktex/utils/lbo.py'
    original = target.read_text()
    try:
        result = patch(original)
    except (ValueError, StopIteration) as exc:
        p.error(str(exc))
    if result == original:
        print('ALREADY PATCHED:', target)
        return
    if not a.apply:
        print(''.join(difflib.unified_diff(original.splitlines(True), result.splitlines(True))))
        print('DRY RUN. Use --apply to install with a backup.')
        return
    backup = target.with_name(target.name + '.before-svd-check-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f') + '.bak')
    shutil.copy2(target, backup)
    if target.read_text() != original:
        raise RuntimeError('Source changed concurrently; refusing to overwrite')
    fd, name = tempfile.mkstemp(prefix='lbo.py.', dir=target.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(result)
        os.chmod(name, target.stat().st_mode)
        os.replace(name, target)
    finally:
        if os.path.exists(name):
            os.unlink(name)
    print('PATCHED:', target)
    print('BACKUP:', backup)


if __name__ == '__main__':
    main()
