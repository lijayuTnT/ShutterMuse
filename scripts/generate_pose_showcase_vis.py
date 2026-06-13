#!/usr/bin/env python3
import argparse
import concurrent.futures
import os
import subprocess
import sys
import time
from pathlib import Path

PROMPT = '我会给出一张keypoints人体骨骼图，请你用一个真人来还原这个骨骼图构成的姿势，要求姿势细节高度一致。注意所有的姿势人物都是正对镜头的。'
DEFAULT_CALL_SCRIPT = Path('/mnt/workspacedir/lijiayu/icons/call_gemini.py')
DEFAULT_INPUT_DIR = Path('/mnt/workspacedir/lijiayu/ShutterMuse-gh-pages/pose_showcase')
DEFAULT_OUTPUT_DIR = Path('/mnt/workspacedir/lijiayu/ShutterMuse-gh-pages/pose_showcase_vis')
DEFAULT_KEEP = {'pose_13', 'pose_17'}


def parse_args():
    parser = argparse.ArgumentParser(description='Generate Subject-side Guidance pose showcase images with GPT-Image-2.')
    parser.add_argument('--input-dir', type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--call-script', type=Path, default=DEFAULT_CALL_SCRIPT)
    parser.add_argument('--workers', type=int, default=5)
    parser.add_argument('--api-key-file', type=Path, default=Path('/tmp/shuttermuse_gpt_key'))
    parser.add_argument('--keep', nargs='*', default=sorted(DEFAULT_KEEP), help='Pose stems to keep and skip, e.g. pose_13 pose_17.')
    parser.add_argument('--size', default='1024x1536')
    parser.add_argument('--input-max-side', default='768')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def get_api_key(args):
    api_key = os.getenv('GPT_API_KEY') or os.getenv('OPENAI_API_KEY')
    if api_key:
        return api_key
    if args.api_key_file.exists():
        return args.api_key_file.read_text(encoding='utf-8').strip()
    raise RuntimeError('Missing GPT_API_KEY/OPENAI_API_KEY or --api-key-file.')


def collect_inputs(input_dir):
    suffixes = {'.jpg', '.jpeg', '.png', '.webp'}
    return sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.name.startswith('pose_') and path.suffix.lower() in suffixes
    )


def cleanup_for_rerun(output_dir, rerun_stems):
    for stem in rerun_stems:
        for path in output_dir.glob(f'{stem}*.png'):
            path.unlink(missing_ok=True)


def generate_one(input_path, output_dir, call_script, api_key, size, input_max_side):
    stem = input_path.stem
    final_output = output_dir / f'{stem}.png'
    tmp_output = output_dir / f'{stem}.tmp.png'
    tmp_output.unlink(missing_ok=True)
    for path in output_dir.glob(f'{stem}.tmp_*.png'):
        path.unlink(missing_ok=True)

    env = os.environ.copy()
    env['GPT_API_KEY'] = api_key
    cmd = [
        sys.executable,
        str(call_script),
        '--backend', 'gpt',
        '--prompt', PROMPT,
        '--image', str(input_path),
        '--output', str(tmp_output),
        '--size', size,
        '--input-max-side', input_max_side,
    ]
    start = time.time()
    proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        tmp_output.unlink(missing_ok=True)
        return stem, False, f'failed code={proc.returncode}\n{proc.stdout[-3000:]}'
    if not tmp_output.exists() or tmp_output.stat().st_size == 0:
        return stem, False, f'missing output\n{proc.stdout[-3000:]}'
    tmp_output.replace(final_output)
    for path in output_dir.glob(f'{stem}.tmp_*.png'):
        path.unlink(missing_ok=True)
    return stem, True, f'{time.time() - start:.1f}s'


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    keep = set(args.keep or [])
    inputs = collect_inputs(args.input_dir)
    rerun_inputs = [path for path in inputs if path.stem not in keep]
    rerun_stems = [path.stem for path in rerun_inputs]

    print(f'total inputs: {len(inputs)}')
    print(f'keep: {sorted(keep)}')
    print(f'rerun: {len(rerun_inputs)}')
    if args.dry_run:
        print('\n'.join(rerun_stems))
        return 0

    api_key = get_api_key(args)
    cleanup_for_rerun(args.output_dir, rerun_stems)

    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                generate_one,
                input_path,
                args.output_dir,
                args.call_script,
                api_key,
                args.size,
                args.input_max_side,
            ): input_path
            for input_path in rerun_inputs
        }
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            completed += 1
            stem, ok, message = future.result()
            status = 'done' if ok else 'error'
            print(f'[{completed:02d}/{len(rerun_inputs):02d}] {stem}: {status} {message}', flush=True)
            if not ok:
                failures.append((stem, message))

    if failures:
        print('\nFailures:')
        for stem, message in failures:
            print(f'--- {stem} ---\n{message}')
        return 1
    print('all rerun images generated successfully')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
