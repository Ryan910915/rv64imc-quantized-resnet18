#!/usr/bin/env python3
"""Build a measurement-only Spike copy from the user's existing clean Git revision.
The original source tree, build directory, and installed Spike are never edited.
"""
from __future__ import annotations
import argparse, datetime, json, os, posixpath, re, shutil, subprocess, sys, tarfile, tempfile
from pathlib import Path, PurePosixPath
from common import sha256

PATCH_ID='resnet-roi-hist-v1-archive-fix1'

def patch_execute(text: str) -> str:
    marker='#include "resnet_profile_mix_observer.h"'
    if marker in text:raise ValueError('Source was already patched; start from a clean revision')
    # Intentionally reject unreviewed variants rather than applying fuzzy patches.
    if 'static inline reg_t execute_insn_logged(' not in text:
        raise ValueError('Unsupported Spike execute.cc layout: execute_insn_logged missing')
    if text.count('npc = fetch.func(p, fetch.insn, pc);')!=1:
        raise ValueError('Unsupported Spike instruction-execution site')
    needle='  p->update_histogram(pc);'
    if text.count(needle)!=1 or '#include <cassert>' not in text:
        raise ValueError('Unsupported Spike histogram hook; do not force this patch')
    a=text.index('static inline reg_t execute_insn_logged(')
    b=text.index(needle)
    if b<a or 'PC_SERIALIZE_BEFORE' not in text[a:b]:
        raise ValueError('Cannot verify post-execution/serialization hook')
    text=text.replace('#include <cassert>','#include <cassert>\n'+marker,1)
    code='''  // Observer counts only successfully retired executions. A serializing CSR
  // can first return PC_SERIALIZE_BEFORE without retiring; do not count it.
  auto& mix = resnet_profile_mix::instance();
  if (mix.enabled) {
    if (npc != PC_SERIALIZE_BEFORE)
      mix.retired(p->get_id(), pc, fetch.insn.bits(), fetch.insn.length());
  } else {
    p->update_histogram(pc);
  }'''
    return text.replace(needle,code,1)

def command(args, **kw):
    print('+',' '.join(map(str,args)),flush=True)
    return subprocess.run(list(map(str,args)),check=True,**kw)

def capture(args,cwd=None):
    return subprocess.check_output(list(map(str,args)),cwd=cwd,text=True).strip()


def extract_git_snapshot(archive: Path, destination: Path) -> None:
    """Extract a local Git archive, including contained relative symlinks.

    Regular files are written before any symlinks are created. Member paths
    cannot traverse a symlink. All links are resolved and checked for containment
    before the staging tree is published. No ownership or privileged mode bits
    are restored. The original repository is never modified.
    """
    if destination.exists() or destination.is_symlink():
        raise ValueError('Snapshot destination already exists: ' + str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.source-extract-', dir=destination.parent))
    try:
        with tarfile.open(archive) as tf:
            members = tf.getmembers()
            paths = {}
            for member in members:
                path = PurePosixPath(member.name)
                if (not member.name or path.is_absolute() or path == PurePosixPath('.')
                        or '..' in path.parts or '\\' in member.name):
                    raise ValueError('Unsafe archive path: ' + member.name)
                if path in paths:
                    raise ValueError('Duplicate archive path: ' + member.name)
                if not (member.isfile() or member.isdir() or member.issym()):
                    raise ValueError('Unsupported archive member: ' + member.name)
                paths[path] = member

            for path, member in paths.items():
                # Do not write files beneath a symlink (regardless of tar order).
                for parent in path.parents:
                    if parent in paths and not paths[parent].isdir():
                        raise ValueError('Archive parent is not a directory: ' + str(parent))
                if member.issym():
                    target = member.linkname
                    if (not target or PurePosixPath(target).is_absolute()
                            or '\\' in target):
                        raise ValueError('Unsafe symbolic-link target: ' + member.name)
                    normalized = posixpath.normpath(str(path.parent / target))
                    if normalized == '..' or normalized.startswith('../'):
                        raise ValueError('Symbolic link leaves snapshot: ' + member.name)

            for path, member in paths.items():
                output = staging.joinpath(*path.parts)
                if member.isdir():
                    output.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    output.parent.mkdir(parents=True, exist_ok=True)
                    data = tf.extractfile(member)
                    if data is None:
                        raise ValueError('Missing archive file data: ' + member.name)
                    with data, output.open('xb') as stream:
                        shutil.copyfileobj(data, stream)
                    output.chmod(0o755 if member.mode & 0o111 else 0o644)

            for path, member in paths.items():
                if member.issym():
                    output = staging.joinpath(*path.parts)
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.symlink_to(member.linkname)

            resolved_root = staging.resolve()
            for path, member in paths.items():
                if member.issym():
                    try:
                        target = staging.joinpath(*path.parts).resolve(strict=True)
                        target.relative_to(resolved_root)
                    except (ValueError, OSError, RuntimeError) as error:
                        raise ValueError('Broken, cyclic, or escaping symbolic link: '
                                         + member.name) from error
            if destination.exists() or destination.is_symlink():
                raise ValueError('Snapshot destination appeared during extraction')
            staging.rename(destination)
    finally:
        # Only this function's newly-created staging directory is removed.
        if staging.exists():
            shutil.rmtree(staging)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--work',type=Path,default=Path('build/mix_spike'))
    ap.add_argument('--jobs',type=int,default=2)
    args=ap.parse_args(); source=args.source.expanduser().resolve();work=args.work.resolve()
    package=Path(__file__).resolve().parent
    if args.jobs<1:raise ValueError('--jobs must be positive')
    if not (source/'riscv/execute.cc').is_file():raise ValueError('Not a Spike source tree: '+str(source))
    dirty=capture(['git','-C',source,'diff','HEAD','--name-only'])
    if dirty:raise ValueError('Spike has tracked local edits. No source was changed. Commit/review those edits before making a pinned snapshot:\n'+dirty)
    commit=capture(['git','-C',source,'rev-parse','HEAD'])
    original=(source/'riscv/execute.cc').read_text();patched=patch_execute(original)
    fingerprint={'patch_id':PATCH_ID,'source_commit':commit,'execute_sha256':sha256(source/'riscv/execute.cc'),
                 'observer_sha256':sha256(package/'observer.h'),'patcher_sha256':sha256(Path(__file__))}
    record=work/'setup.json'
    if record.exists():
        previous=json.loads(record.read_text())
        if previous['fingerprint']!=fingerprint:
            raise ValueError('This work directory belongs to another source/patch version. Choose a NEW --work directory; old evidence was not removed.')
        if previous.get('complete') and (work/'install/bin/spike').is_file():
            print('Instrumented Spike already built:',work/'install/bin/spike');return 0
    elif work.exists() and any(work.iterdir()):
        raise ValueError('Nonempty work directory without a setup record. Choose a NEW --work directory.')
    work.mkdir(parents=True,exist_ok=True)
    record.write_text(json.dumps({'fingerprint':fingerprint,'source':str(source),'complete':False},indent=2)+'\n')
    snapshot=work/'source'; build=work/'build';install=work/'install'
    if not snapshot.exists():
        archive=work/'source.tar'
        command(['git','-C',source,'archive','--format=tar','--output',archive,'HEAD'])
        extract_git_snapshot(archive, snapshot)
        archive.unlink()
        (snapshot/'riscv/execute.cc').write_text(patched)
        shutil.copy2(package/'observer.h',snapshot/'riscv/resnet_profile_mix_observer.h')
    if (snapshot/'riscv/execute.cc').read_text()!=patched:
        raise ValueError('Snapshot differs from expected patched source; refusing to continue')
    build.mkdir(exist_ok=True)
    env=os.environ.copy();env['CC']='/usr/bin/gcc';env['CXX']='/usr/bin/g++'
    env.pop('CONDA_BUILD_SYSROOT',None)
    if not (build/'Makefile').exists():
        command([snapshot/'configure','--prefix='+str(install),'CC=/usr/bin/gcc','CXX=/usr/bin/g++'],cwd=build,env=env)
    command(['make',f'-j{args.jobs}'],cwd=build,env=env)
    command(['make','install'],cwd=build,env=env)
    record.write_text(json.dumps({'fingerprint':fingerprint,'source':str(source),'complete':True,
        'binary_sha256':sha256(install/'bin/spike'),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2)+'\n')
    print('READY:',install/'bin/spike')
    print('Original Spike installation and original source tree are unchanged.')
    return 0
if __name__=='__main__':
    try:sys.exit(main())
    except (ValueError,OSError,subprocess.CalledProcessError) as e:
        print('SETUP STOPPED:',e,file=sys.stderr);sys.exit(2)
