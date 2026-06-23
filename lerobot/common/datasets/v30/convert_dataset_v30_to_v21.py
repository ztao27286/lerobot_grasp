#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Convert a local LeRobot dataset from codebase version v3.0 back to v2.1.

This is the reverse of `convert_dataset_v21_to_v30.py` for local datasets. It:

- Splits v3.0 data shards (`data/chunk-xxx/file-xxx.parquet`) back into one
  parquet file per episode.
- Converts parquet metadata (`meta/tasks.parquet`,
  `meta/episodes/chunk-xxx/file-xxx.parquet`) back to v2.1 JSONL metadata.
- Recreates `meta/episodes_stats.jsonl` from the per-episode stats embedded in
  the v3.0 episode metadata.
- Splits consolidated v3.0 videos back into one mp4 per episode.
- Writes a v2.1-style `meta/info.json`.

Example:

```bash
python src/lerobot/datasets/v30/convert_dataset_v30_to_v21.py \
    --input-dir /path/to/v30_dataset \
    --output-dir /path/to/v21_dataset
```
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import shutil
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is a LeRobot dependency, this keeps --help usable.
    tqdm = lambda iterable, **_: iterable


V30 = "v3.0"
V21 = "v2.1"

DEFAULT_CHUNK_SIZE = 1000

INFO_PATH = Path("meta/info.json")
V30_TASKS_PATH = Path("meta/tasks.parquet")
V30_EPISODES_DIR = Path("meta/episodes")

V21_EPISODES_PATH = Path("meta/episodes.jsonl")
V21_EPISODES_STATS_PATH = Path("meta/episodes_stats.jsonl")
V21_TASKS_PATH = Path("meta/tasks.jsonl")
V21_DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
V21_VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"

DEFAULT_V30_DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
DEFAULT_V30_VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def write_jsonlines(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(to_jsonable(record), ensure_ascii=False))
            f.write("\n")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def unflatten_dict(flat: dict[str, Any], sep: str = "/") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in flat.items():
        parts = key.split(sep)
        node = out
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return out


def format_path(template: str, **kwargs: Any) -> Path:
    try:
        return Path(template.format(**kwargs))
    except KeyError as exc:
        raise KeyError(f"Path template {template!r} is missing key {exc!s}") from exc


def normalize_task_list(value: Any, task_index_to_task: dict[int, str]) -> list[str]:
    value = to_jsonable(value)
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (int, np.integer)):
        return [task_index_to_task.get(int(value), str(value))]
    if not isinstance(value, list):
        return [str(value)]

    tasks: list[str] = []
    for item in value:
        if isinstance(item, (int, np.integer)):
            tasks.append(task_index_to_task.get(int(item), str(item)))
        else:
            tasks.append(str(item))
    return tasks


def validate_input_dataset(input_dir: Path, force: bool) -> dict[str, Any]:
    info_path = input_dir / INFO_PATH
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing {INFO_PATH} in input dataset: {input_dir}")

    info = load_json(info_path)
    version = info.get("codebase_version")
    if version != V30 and not force:
        raise ValueError(
            f"Input dataset codebase_version is {version!r}, expected {V30!r}. "
            "Use --force if you intentionally want to try converting it anyway."
        )
    return info


def load_v30_episodes(input_dir: Path) -> pd.DataFrame:
    paths = sorted((input_dir / V30_EPISODES_DIR).glob("chunk-*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No v3.0 episode metadata parquet files found under {input_dir / V30_EPISODES_DIR}")

    episodes = pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)
    if "episode_index" not in episodes:
        raise ValueError("Episode metadata is missing required column 'episode_index'.")

    episodes = episodes.sort_values("episode_index").reset_index(drop=True)
    episode_indices = episodes["episode_index"].astype(int).tolist()
    expected_indices = list(range(len(episode_indices)))
    if episode_indices != expected_indices:
        raise ValueError(
            "This converter expects contiguous episode_index values starting at 0. "
            f"Found {episode_indices[:10]}..."
        )
    return episodes


def load_v30_tasks(input_dir: Path) -> tuple[list[dict[str, Any]], dict[int, str]]:
    tasks_path = input_dir / V30_TASKS_PATH
    if not tasks_path.is_file():
        raise FileNotFoundError(f"Missing v3.0 tasks file: {tasks_path}")

    tasks_df = pd.read_parquet(tasks_path)
    if "task_index" in tasks_df:
        task_indices = tasks_df["task_index"].astype(int).tolist()
    else:
        task_indices = list(range(len(tasks_df)))

    if "task" in tasks_df:
        task_strings = tasks_df["task"].astype(str).tolist()
    elif "__index_level_0__" in tasks_df:
        task_strings = tasks_df["__index_level_0__"].astype(str).tolist()
    else:
        task_strings = [str(item) for item in tasks_df.index.tolist()]

    records = [
        {"task_index": int(task_index), "task": task}
        for task_index, task in sorted(zip(task_indices, task_strings, strict=True), key=lambda item: item[0])
    ]
    return records, {record["task_index"]: record["task"] for record in records}


def stats_from_episode_row(row: pd.Series) -> dict[str, Any]:
    stats_flat = {
        key: to_jsonable(value)
        for key, value in row.items()
        if isinstance(key, str) and key.startswith("stats/")
    }
    if not stats_flat:
        raise ValueError(
            f"Episode {int(row['episode_index'])} has no 'stats/...' columns in v3.0 metadata."
        )
    return unflatten_dict(stats_flat)["stats"]


def write_v21_metadata(
    episodes: pd.DataFrame,
    tasks: list[dict[str, Any]],
    task_index_to_task: dict[int, str],
    output_dir: Path,
) -> None:
    write_jsonlines(tasks, output_dir / V21_TASKS_PATH)

    episode_records: list[dict[str, Any]] = []
    stats_records: list[dict[str, Any]] = []
    for _, row in episodes.iterrows():
        ep_idx = int(row["episode_index"])
        episode_records.append(
            {
                "episode_index": ep_idx,
                "tasks": normalize_task_list(row["tasks"], task_index_to_task),
                "length": int(row["length"]),
            }
        )
        stats_records.append({"episode_index": ep_idx, "stats": stats_from_episode_row(row)})

    write_jsonlines(episode_records, output_dir / V21_EPISODES_PATH)
    write_jsonlines(stats_records, output_dir / V21_EPISODES_STATS_PATH)


def make_v21_info(info: dict[str, Any], episodes: pd.DataFrame, tasks: list[dict[str, Any]]) -> dict[str, Any]:
    out_info = copy.deepcopy(info)
    chunks_size = int(out_info.get("chunks_size", DEFAULT_CHUNK_SIZE))
    total_episodes = int(len(episodes))
    total_frames = int(episodes["length"].astype(int).sum())
    video_keys = get_video_keys(out_info)

    out_info["codebase_version"] = V21
    out_info["total_episodes"] = total_episodes
    out_info["total_frames"] = total_frames
    out_info["total_tasks"] = len(tasks)
    out_info["total_videos"] = total_episodes * len(video_keys)
    out_info["total_chunks"] = math.ceil(total_episodes / chunks_size) if total_episodes else 0
    out_info["chunks_size"] = chunks_size
    out_info["splits"] = {"train": f"0:{total_episodes}"}
    out_info["data_path"] = V21_DATA_PATH
    out_info["video_path"] = V21_VIDEO_PATH if video_keys else None

    out_info.pop("data_files_size_in_mb", None)
    out_info.pop("video_files_size_in_mb", None)

    for feature in out_info.get("features", {}).values():
        if feature.get("dtype") != "video":
            feature.pop("fps", None)

    fps = out_info.get("fps")
    if isinstance(fps, float) and fps.is_integer():
        out_info["fps"] = int(fps)

    return out_info


def get_video_keys(info: dict[str, Any]) -> list[str]:
    return sorted(
        key for key, feature in info.get("features", {}).items() if feature.get("dtype") == "video"
    )


def get_chunks_size(info: dict[str, Any]) -> int:
    return int(info.get("chunks_size", DEFAULT_CHUNK_SIZE))


def get_v30_data_file(info: dict[str, Any], chunk_index: int, file_index: int) -> Path:
    template = info.get("data_path") or DEFAULT_V30_DATA_PATH
    return format_path(template, chunk_index=chunk_index, file_index=file_index)


def get_v30_video_file(info: dict[str, Any], video_key: str, chunk_index: int, file_index: int) -> Path:
    template = info.get("video_path") or DEFAULT_V30_VIDEO_PATH
    return format_path(
        template,
        video_key=video_key,
        chunk_index=chunk_index,
        file_index=file_index,
    )


def get_v21_data_file(episode_index: int, chunks_size: int) -> Path:
    return format_path(
        V21_DATA_PATH,
        episode_chunk=episode_index // chunks_size,
        episode_index=episode_index,
    )


def get_v21_video_file(episode_index: int, video_key: str, chunks_size: int) -> Path:
    return format_path(
        V21_VIDEO_PATH,
        episode_chunk=episode_index // chunks_size,
        video_key=video_key,
        episode_index=episode_index,
    )


def filter_episode_table(table: pa.Table, row: pd.Series, file_start_index: int) -> pa.Table:
    ep_idx = int(row["episode_index"])
    expected_length = int(row["length"])

    if "episode_index" in table.column_names:
        filtered = table.filter(pc.equal(table["episode_index"], ep_idx))
        if filtered.num_rows == expected_length:
            return filtered

    local_start = int(row["dataset_from_index"]) - file_start_index
    sliced = table.slice(local_start, expected_length)
    if sliced.num_rows != expected_length:
        raise ValueError(
            f"Could not recover episode {ep_idx}: expected {expected_length} rows, "
            f"got {sliced.num_rows} rows."
        )
    return sliced


def convert_data_files(episodes: pd.DataFrame, info: dict[str, Any], input_dir: Path, output_dir: Path) -> None:
    chunks_size = get_chunks_size(info)
    group_cols = ["data/chunk_index", "data/file_index"]
    for (chunk_idx, file_idx), group in tqdm(
        episodes.groupby(group_cols, sort=True),
        desc="Splitting data parquet files",
        total=episodes[group_cols].drop_duplicates().shape[0],
    ):
        rel_path = get_v30_data_file(info, int(chunk_idx), int(file_idx))
        src_path = input_dir / rel_path
        if not src_path.is_file():
            raise FileNotFoundError(f"Missing v3.0 data parquet file: {src_path}")

        table = pq.read_table(src_path)
        file_start_index = int(group["dataset_from_index"].min())

        for _, row in group.sort_values("episode_index").iterrows():
            ep_idx = int(row["episode_index"])
            ep_table = filter_episode_table(table, row, file_start_index)
            dst_path = output_dir / get_v21_data_file(ep_idx, chunks_size)
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(ep_table, dst_path, compression="snappy")


def get_video_pixel_channels(pix_fmt: str) -> int:
    if "gray" in pix_fmt or "depth" in pix_fmt or "monochrome" in pix_fmt:
        return 1
    if "rgba" in pix_fmt or "yuva" in pix_fmt:
        return 4
    if "rgb" in pix_fmt or "yuv" in pix_fmt:
        return 3
    raise ValueError(f"Unknown video pixel format: {pix_fmt}")


def get_video_info(video_path: Path) -> dict[str, Any]:
    import av

    logging.getLogger("libav").setLevel(av.logging.ERROR)
    video_info: dict[str, Any] = {}
    with av.open(str(video_path), "r") as video_file:
        try:
            video_stream = video_file.streams.video[0]
        except IndexError:
            return {}

        video_info["video.height"] = video_stream.height
        video_info["video.width"] = video_stream.width
        video_info["video.codec"] = video_stream.codec.canonical_name
        video_info["video.pix_fmt"] = video_stream.pix_fmt
        video_info["video.is_depth_map"] = False
        video_info["video.fps"] = int(video_stream.base_rate)
        video_info["video.channels"] = get_video_pixel_channels(video_stream.pix_fmt)

        try:
            audio_stream = video_file.streams.audio[0]
        except IndexError:
            video_info["has_audio"] = False
        else:
            video_info["audio.channels"] = audio_stream.channels
            video_info["audio.codec"] = audio_stream.codec.canonical_name
            video_info["audio.bit_rate"] = audio_stream.bit_rate
            video_info["audio.sample_rate"] = audio_stream.sample_rate
            video_info["audio.bit_depth"] = audio_stream.format.bits
            video_info["audio.channel_layout"] = audio_stream.layout.name
            video_info["has_audio"] = True

    av.logging.restore_default_callback()
    return video_info


def extract_video_segment(
    src_path: Path,
    dst_path: Path,
    start_s: float,
    end_s: float,
    expected_frames: int,
    fps: int,
    vcodec: str,
) -> None:
    import av

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst_path.with_suffix(".tmp.mp4")
    if tmp_path.exists():
        tmp_path.unlink()

    logging.getLogger("libav").setLevel(av.logging.ERROR)
    with av.open(str(src_path), "r") as input_container:
        input_stream = input_container.streams.video[0]
        rate = int(input_stream.base_rate) if input_stream.base_rate else int(fps)
        rate = rate or int(fps)
        frame_tol = 0.5 / rate
        time_base = Fraction(1, rate)

        try:
            seek_offset = int(max(start_s - 1.0, 0.0) / float(input_stream.time_base))
            input_container.seek(seek_offset, any_frame=False, backward=True, stream=input_stream)
        except Exception:
            input_container.seek(int(max(start_s - 1.0, 0.0) * av.time_base), any_frame=False, backward=True)

        video_options: dict[str, str] = {"g": "2"}
        if vcodec in {"h264", "hevc"}:
            video_options["crf"] = "30"
            video_options["bf"] = "0"
        elif vcodec == "libsvtav1":
            video_options["crf"] = "30"
            video_options["preset"] = "12"

        with av.open(str(tmp_path), "w", options={"movflags": "faststart"}) as output_container:
            output_stream = output_container.add_stream(vcodec, rate, options=video_options)
            output_stream.width = input_stream.width
            output_stream.height = input_stream.height
            output_stream.pix_fmt = "yuv420p"
            output_stream.time_base = time_base
            output_stream.codec_context.time_base = time_base

            written = 0
            for frame in input_container.decode(input_stream):
                if frame.pts is None:
                    timestamp = written / rate
                else:
                    timestamp = float(frame.pts * frame.time_base)

                if timestamp + frame_tol < start_s:
                    continue
                if timestamp > end_s + frame_tol:
                    break
                if written >= expected_frames:
                    break

                out_frame = frame.reformat(
                    width=input_stream.width,
                    height=input_stream.height,
                    format=output_stream.pix_fmt,
                )
                out_frame.pts = written
                out_frame.time_base = time_base
                for packet in output_stream.encode(out_frame):
                    output_container.mux(packet)
                written += 1

            for packet in output_stream.encode():
                output_container.mux(packet)

    av.logging.restore_default_callback()

    if written == 0:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"No frames were written while extracting {src_path} [{start_s}, {end_s}]")

    shutil.move(str(tmp_path), str(dst_path))


def convert_videos(
    episodes: pd.DataFrame,
    info: dict[str, Any],
    input_dir: Path,
    output_dir: Path,
    skip_videos: bool,
    vcodec: str,
) -> None:
    video_keys = get_video_keys(info)
    if not video_keys:
        return

    if skip_videos:
        logging.warning("Skipping video conversion; output metadata will still declare video features.")
        return

    chunks_size = get_chunks_size(info)
    fps = int(info["fps"])
    for video_key in video_keys:
        required_cols = [
            f"videos/{video_key}/chunk_index",
            f"videos/{video_key}/file_index",
            f"videos/{video_key}/from_timestamp",
            f"videos/{video_key}/to_timestamp",
        ]
        missing = [col for col in required_cols if col not in episodes]
        if missing:
            raise ValueError(f"Episode metadata is missing video columns for {video_key}: {missing}")

        group_cols = [f"videos/{video_key}/chunk_index", f"videos/{video_key}/file_index"]
        for (chunk_idx, file_idx), group in tqdm(
            episodes.groupby(group_cols, sort=True),
            desc=f"Splitting videos for {video_key}",
            total=episodes[group_cols].drop_duplicates().shape[0],
        ):
            rel_path = get_v30_video_file(info, video_key, int(chunk_idx), int(file_idx))
            src_path = input_dir / rel_path
            if not src_path.is_file():
                raise FileNotFoundError(f"Missing v3.0 video file: {src_path}")

            sorted_group = group.sort_values("episode_index")
            can_copy_whole_file = len(sorted_group) == 1
            for _, row in sorted_group.iterrows():
                ep_idx = int(row["episode_index"])
                dst_path = output_dir / get_v21_video_file(ep_idx, video_key, chunks_size)
                start_s = float(row[f"videos/{video_key}/from_timestamp"])
                end_s = float(row[f"videos/{video_key}/to_timestamp"])
                expected_frames = int(row["length"])

                if can_copy_whole_file and abs(start_s) < 1e-6:
                    dst_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_path, dst_path)
                else:
                    extract_video_segment(src_path, dst_path, start_s, end_s, expected_frames, fps, vcodec)


def refresh_video_info(info: dict[str, Any], output_dir: Path) -> None:
    video_keys = get_video_keys(info)
    if not video_keys:
        return

    chunks_size = get_chunks_size(info)
    for video_key in video_keys:
        first_video = output_dir / get_v21_video_file(0, video_key, chunks_size)
        if first_video.is_file():
            info["features"][video_key]["info"] = get_video_info(first_video)


def copy_dataset_card(input_dir: Path, output_dir: Path) -> None:
    for name in ("README.md", ".gitattributes"):
        src = input_dir / name
        if src.is_file():
            shutil.copy2(src, output_dir / name)


def validate_output_counts(
    output_dir: Path,
    info: dict[str, Any],
    skip_videos: bool,
) -> None:
    total_episodes = int(info["total_episodes"])
    data_files = sorted((output_dir / "data").glob("chunk-*/*.parquet"))
    if len(data_files) != total_episodes:
        raise RuntimeError(f"Expected {total_episodes} data parquet files, found {len(data_files)}.")

    if skip_videos:
        return

    video_keys = get_video_keys(info)
    for video_key in video_keys:
        video_files = sorted((output_dir / "videos").glob(f"chunk-*/{video_key}/episode_*.mp4"))
        if len(video_files) != total_episodes:
            raise RuntimeError(
                f"Expected {total_episodes} videos for {video_key}, found {len(video_files)}."
            )


def prepare_output_dir(input_dir: Path, output_dir: Path, overwrite: bool) -> None:
    input_resolved = input_dir.resolve()
    output_resolved = output_dir.resolve()
    if input_resolved == output_resolved:
        raise ValueError("In-place downgrade is not supported. Please choose a different --output-dir.")

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}. Use --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def convert_dataset(
    input_dir: str | Path,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
    force: bool = False,
    skip_videos: bool = False,
    vcodec: str = "h264",
) -> Path:
    input_path = Path(input_dir).expanduser().resolve()
    output_path = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else input_path.with_name(f"{input_path.name}_v21")
    )

    info = validate_input_dataset(input_path, force)
    episodes = load_v30_episodes(input_path)
    tasks, task_index_to_task = load_v30_tasks(input_path)
    out_info = make_v21_info(info, episodes, tasks)

    prepare_output_dir(input_path, output_path, overwrite)

    logging.info("Writing v2.1 metadata")
    write_v21_metadata(episodes, tasks, task_index_to_task, output_path)

    logging.info("Splitting v3.0 data shards")
    convert_data_files(episodes, info, input_path, output_path)

    logging.info("Splitting v3.0 video shards")
    convert_videos(episodes, info, input_path, output_path, skip_videos, vcodec)
    if not skip_videos:
        refresh_video_info(out_info, output_path)

    logging.info("Writing v2.1 info.json")
    write_json(out_info, output_path / INFO_PATH)

    copy_dataset_card(input_path, output_path)
    validate_output_counts(output_path, out_info, skip_videos)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Path to the local v3.0 dataset directory that contains meta/info.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Path to write the v2.1 dataset. Defaults to a sibling named '<input>_v21'.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace --output-dir if it already exists.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Try conversion even if meta/info.json is not marked as codebase_version v3.0.",
    )
    parser.add_argument(
        "--skip-videos",
        action="store_true",
        help="Only convert parquet data and metadata. Use this for a quick metadata/data test.",
    )
    parser.add_argument(
        "--vcodec",
        type=str,
        default="h264",
        choices=["h264", "hevc", "libsvtav1"],
        help="Codec used when splitting multi-episode v3.0 videos. Defaults to h264.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    args = parse_args()
    output_path = convert_dataset(**vars(args))
    print(f"Converted dataset written to: {output_path}")


if __name__ == "__main__":
    main()
