import logging

import click

from src.utils import preprocess
from src.eval import generate
from src.utils.mesh_utils import meshcleaning

logger = logging.getLogger(__name__)

context_settings = {
    "help_option_names": ["-h", "--help"],
    "show_default": True,
    "ignore_unknown_options": True,
    "allow_extra_args": True,
}

@click.command(context_settings=context_settings)
@click.option(
    "-i",
    "--input_video",
    required=True,
    help="Input MP4 file",
)
@click.option(
    "-o",
    "--output_dir",
    type=click.Path(file_okay=False, writable=True),
    required=True,
    help="Directory to store OBJ files.",
)
@click.option(
    "-r",
    "--resolution",
    default=256,
    help="Resolution",
)
@click.option(
    "-f",
    "--fps",
    default=24,
    help="Sampling FPS from video (This is not the input video fps)",
)
@click.option(
    "-g",
    "--gpu_id",
    default=0,
    help="Sampling FPS from video (This is not the input video fps)",
)
@click.option(
    "-c",
    "--meshclean/--no-meshclean",
    default=False,
)
def cli(input_video, output_dir, resolution, fps, gpu_id, meshclean, generate_unity_data=True):
    ctx = click.get_current_context()
    # Convert to frames
    data_dir = preprocess(input_video, fps=fps)
    # data_dir = "__temp_vid_frames_input"

    # Predict & Save series of obj
    results_dir, ok = generate(
        ckpt_path="checkpoints/pifuhd.pt",
        dataroot=data_dir,
        resolution=resolution,
        results_path=output_dir,
        load_size=1024,
        use_rect=True,
        gpu_id=gpu_id)

    # results_dir, ok = '%s/%s/recon' % (output_dir, 'pifuhd_final'), True
    
    if ok and meshclean:
        meshcleaning(results_dir)
    
    # if generate_unity_data:
        # Convert to dae for Unity consumption


if __name__ == "__main__":
    cli()


