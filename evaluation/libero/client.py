import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import imageio
import numpy as np
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from lerobot.datasets.utils import write_json
from tqdm import tqdm

from wan_va.utils.Simple_Remote_Infer.deploy.websocket_client_policy import WebsocketClientPolicy


CAM_KEYS = [
    "observation.images.agentview_rgb",
    "observation.images.eye_in_hand_rgb",
]
PRED_VIDEO_RE = re.compile(r"^pred_video_(\d+)\.mp4$")


def save_video(real_obs_list, save_path, fps=15, video_names=CAM_KEYS):
    if not real_obs_list:
        print("❌ No real observation frames")
        return

    first_obs = real_obs_list[0]
    base_h, width_base = first_obs[video_names[0]].shape[:2]
    target_size = (width_base, base_h)

    print(f"Saving video: {len(real_obs_list)} frames...")

    final_frames = [
        np.hstack([cv2.resize(obs[name], target_size) for name in video_names]).astype(np.uint8)
        for obs in real_obs_list
    ]

    save_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(save_path, final_frames, fps=fps)
    print(f"✅ Video saved to: {save_path}")


def construct_single_env(env_args):
    count = 0
    env = None
    env_creation = False
    while not env_creation and count < 5:
        try:
            env = OffScreenRenderEnv(**env_args)
            env_creation = True
        except Exception as e:
            print(f"Error!!!  construct env failed: {e}")
            time.sleep(5)
            count += 1
    if count >= 5:
        return None
    return env


def _extract_obs(obs):
    agentview = np.ascontiguousarray(obs["agentview_image"][::-1])
    eye_in_hand = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1])
    return {"observation.images.agentview_rgb": agentview, "observation.images.eye_in_hand_rgb": eye_in_hand}


def init_single_env(env_in, init_state):
    env_in.reset()
    env_in.set_init_state(init_state)
    for _ in range(5):
        obs, _, _, _ = env_in.step([0.] * 7)
    return _extract_obs(obs)


def env_one_step(env_in, action):
    obs, _, done, _ = env_in.step(action)
    return _extract_obs(obs), done


def resolve_shared_path(path_value, out_dir):
    if not path_value:
        return None
    path = Path(path_value)
    if path.exists():
        return path
    parts = path.parts
    for anchor in ("visualization", "libero_eval", "robomme_eval"):
        if anchor in parts:
            idx = parts.index(anchor)
            candidate = out_dir.joinpath(*parts[idx:])
            if candidate.exists():
                return candidate
    return path


def copy_prediction_video(server_video_path, out_dir, run_dir):
    src = resolve_shared_path(server_video_path, out_dir)
    if src is None or not src.exists():
        if server_video_path:
            print(f"[warn] prediction video not found: {server_video_path}")
        return None
    dst = run_dir / src.name
    try:
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
    except Exception as exc:
        print(f"[warn] failed to copy {src} -> {dst}: {exc}")
        return None
    return dst.name


def copy_prediction_videos_from_visualization(visualization_dir, out_dir, run_dir, seen):
    vis_dir = resolve_shared_path(visualization_dir, out_dir)
    if vis_dir is None or not vis_dir.exists():
        return []
    copied = []
    for src in sorted(vis_dir.glob("pred_video_*.mp4")):
        if src.name in seen:
            continue
        name = copy_prediction_video(src, out_dir, run_dir)
        if name is not None:
            seen.add(name)
            copied.append(name)
    return copied


def _ffconcat_quote(path):
    text = str(path.resolve())
    return "'" + text.replace("\\", "\\\\").replace("'", "'\\''") + "'"


def concat_prediction_videos(run_dir, output_name="pred_video_all.mp4"):
    videos = [p for p in run_dir.glob("pred_video_*.mp4") if PRED_VIDEO_RE.match(p.name)]
    if not videos:
        return None
    videos.sort(key=lambda p: int(PRED_VIDEO_RE.match(p.name).group(1)))
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("[warn] ffmpeg not found; skip pred_video_all.mp4")
        return None

    output_path = run_dir / output_name
    tmp_output = output_path.with_name(output_path.name + ".tmp.mp4")
    list_path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".ffconcat", delete=False, encoding="utf-8") as handle:
            list_path = Path(handle.name)
            for video in videos:
                handle.write(f"file {_ffconcat_quote(video)}\n")
        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "concat", "-safe", "0", "-i", str(list_path),
            "-c", "copy", str(tmp_output),
        ]
        subprocess.run(cmd, check=True)
        os.replace(tmp_output, output_path)
        return output_path.name
    except Exception as exc:
        print(f"[warn] failed to concat prediction videos in {run_dir}: {exc}")
        return None
    finally:
        if list_path is not None:
            list_path.unlink(missing_ok=True)
        tmp_output.unlink(missing_ok=True)


def run_one(model, libero_benchmark, task_idx, out_dir, episode_idx, save_pred_videos=True):
    benchmark_dict = benchmark.get_benchmark_dict()
    benchmark_instance = benchmark_dict[libero_benchmark]()
    num_tasks = benchmark_instance.get_num_tasks()
    assert task_idx < num_tasks, f"Error: error id must smaller than {num_tasks}"
    prompt = benchmark_instance.get_task(task_idx).language
    env_args = {
                "bddl_file_name": benchmark_instance.get_task_bddl_file_path(task_idx),
                "camera_heights": 128,
                "camera_widths": 128,
            }
    init_states = benchmark_instance.get_task_init_states(task_idx)

    cur_env = construct_single_env(env_args)
    first_obs = init_single_env(cur_env, init_states[episode_idx % init_states.shape[0]])

    task_dir_name = f"{task_idx}_{prompt.replace(' ', '_')}"
    run_dir = Path(out_dir) / libero_benchmark / task_dir_name / f"ep{episode_idx:04d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "instruction.txt").write_text(prompt + "\n", encoding="utf-8")

    vis_kwargs = {
        "task_name": f"{libero_benchmark}_{task_idx}",
        "env_id": f"{libero_benchmark}_{task_idx}",
        "save_visualization": save_pred_videos,
    }

    reset_meta = model.infer(dict(reset=True, prompt=prompt, **vis_kwargs))
    if not isinstance(reset_meta, dict):
        reset_meta = {}

    full_obs_list = []
    done = False
    first = True
    copied_pred_names = set()
    while cur_env.env.timestep < 800:
        ret = model.infer(dict(obs=first_obs, prompt=prompt, **vis_kwargs))
        if save_pred_videos and isinstance(ret, dict):
            name = copy_prediction_video(ret.get("video_path"), Path(out_dir), run_dir)
            if name is not None:
                copied_pred_names.add(name)
        action = ret['action']

        key_frame_list = []
        assert action.shape[2] % 4 == 0
        action_per_frame = action.shape[2] // 4
        start_idx = 1 if first else 0
        for i in range(start_idx, action.shape[1]):
            for j in range(action.shape[2]):
                ee_action = action[:, i, j]
                observes, done = env_one_step(cur_env, ee_action)
                if done:
                    break
                if (j+1) % action_per_frame == 0:
                    full_obs_list.append(observes)
                    key_frame_list.append(observes)

            if done:
                break

        first = False

        if done:
            break
        else:
            model.infer(dict(obs=key_frame_list, compute_kv_cache=True, imagine=False, state=action, **vis_kwargs))

    pred_video_all = None
    if save_pred_videos:
        copy_prediction_videos_from_visualization(
            reset_meta.get("visualization_dir"),
            Path(out_dir),
            run_dir,
            copied_pred_names,
        )
        pred_video_all = concat_prediction_videos(run_dir)

    out_file = run_dir / f"{episode_idx}_{done}.mp4"
    save_video(
        real_obs_list=full_obs_list,
        save_path=out_file,
        fps=60,
        video_names=CAM_KEYS,
    )

    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "benchmark": libero_benchmark,
                "task_idx": task_idx,
                "episode_idx": episode_idx,
                "instruction": prompt,
                "success": bool(done),
                "client_video": out_file.name,
                "prediction_video_all": pred_video_all,
                "server_visualization_dir": reset_meta.get("visualization_dir"),
                "server_run_name": reset_meta.get("run_name"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    cur_env.close()
    return done


def _episode_success_from_run_dir(run_dir: Path) -> bool | None:
    for mp4 in run_dir.glob("*_True.mp4"):
        if mp4.name != "pred_video_all.mp4":
            return True
    for mp4 in run_dir.glob("*_False.mp4"):
        if mp4.name != "pred_video_all.mp4":
            return False
    return None


def run(libero_benchmark, port, out_dir, test_num, task_range=None, save_pred_videos=True):
    if task_range is None:
        benchmark_dict = benchmark.get_benchmark_dict()
        benchmark_instance = benchmark_dict[libero_benchmark]()
        num_tasks = benchmark_instance.get_num_tasks()
        progress_bar = tqdm(range(num_tasks), total=num_tasks)
    else:
        assert len(task_range) == 2, f'task_range: [start, end) for splitting tasks, however, task_range: {task_range}'
        num_tasks = task_range[1] - task_range[0]
        progress_bar = tqdm(range(task_range[0], task_range[1]), total=num_tasks)

    print(f"#################### Use benchmark: {libero_benchmark}, num_tasks: {num_tasks} #############")
    model = WebsocketClientPolicy(port=port)

    episode_list = range(test_num)
    for task_idx in progress_bar:
        succ_num = 0.
        for episode_idx in tqdm(episode_list, total=len(episode_list)):
            res_i = run_one(
                model, libero_benchmark, task_idx, out_dir, episode_idx,
                save_pred_videos=save_pred_videos,
            )
            succ_num += res_i
            succ_rate = succ_num / (episode_idx + 1)
            print(f"Success rate: {succ_rate}, success num: {succ_num}, total num: {episode_idx + 1}")
            out_file = Path(out_dir) / f"{libero_benchmark}_{task_idx}.json"
            out_file.parent.mkdir(exist_ok=True, parents=True)
            write_json({
                "succ_num": succ_num,
                "total_num": episode_idx + 1.,
                "succ_rate": succ_rate,
                }, out_file
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--libero-benchmark",
        type=str,
        default="libero_10",
        choices=["libero_10", "libero_goal", "libero_spatial", "libero_object"],
        help="Benchmark name",
    )
    parser.add_argument(
        "--task-range",
        type=int,
        nargs="+",
        default=[0, 10],
        help="Task range [start, end) for splitting tasks",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=23908,
        help="WebSocket port",
    )
    parser.add_argument(
        "--test-num",
        type=int,
        default=50,
        help="Number of test episodes",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/libero",
        help="Output directory for results",
    )
    parser.add_argument(
        "--no-pred-videos",
        action="store_true",
        help="Disable saving server-side predicted videos (pred_video_*.mp4 and pred_video_all.mp4).",
    )
    args = parser.parse_args()
    run(
        libero_benchmark=args.libero_benchmark,
        port=args.port,
        out_dir=args.out_dir,
        test_num=args.test_num,
        task_range=args.task_range,
        save_pred_videos=not args.no_pred_videos,
    )
    print("Finish all process!!!!!!!!!!!!")


if __name__ == "__main__":
    main()
