"""Patch a pinned upstream Wan VACE example for the Absent 17-frame trial."""

from pathlib import Path
import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("upstream", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("trial_dir", type=Path)
    args = parser.parse_args()
    source = args.upstream.read_text()
    replacements = {
        'GPU_memory_mode     = "sequential_cpu_offload"': 'GPU_memory_mode     = "model_full_load"',
        'fsdp_text_encoder   = True': 'fsdp_text_encoder   = False',
        'video_length        = 81': 'video_length        = 17',
        'inpaint_video       = "asset/inpaint_video.mp4"': f'inpaint_video       = "{args.trial_dir}/source.mp4"',
        'inpaint_video_mask  = "asset/inpaint_video_mask.mp4"': f'inpaint_video_mask  = "{args.trial_dir}/mask.mp4"',
        'num_inference_steps     = 50': 'num_inference_steps     = 30',
        'save_path               = "samples/vace-videos-fun"': f'save_path               = "{args.trial_dir}/output"',
        'prompt              = "一只棕色的兔子舔了一下它的舌头，坐在舒适房间里的浅色沙发上。在兔子的后面，架子上有一幅镶框的画，周围是粉红色的花朵。房间里柔和温暖的灯光营造出舒适的氛围。"': 'prompt              = "An empty dining room. Preserve the table, dishes, glasses, chairs, walls, lighting, and camera viewpoint. No people."',
        'negative_prompt     = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"': 'negative_prompt     = "people, person, face, body, hands, silhouette, distorted furniture, moving camera, text, low quality"',
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise RuntimeError(f"Expected one upstream setting, found {source.count(old)}: {old[:80]}")
        source = source.replace(old, new, 1)
    args.output.write_text(source)
    print(f"Configured {args.output}")


if __name__ == "__main__":
    main()
