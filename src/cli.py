import logging

import click

from src.utils import preprocess

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
    "--fps",
    required=False,
    default=24,
    help="Sampling FPS from video (This is not the input video fps)",
)
def cli(input_video, output_dir, fps, generate_unity_data=True):
    ctx = click.get_current_context()
    
    # Convert to frames
    data_dir = preprocess(input_video)

    # Predict & Save series of obj
    print(data_dir)
    
    # if generate_unity_data:
        # Convert to dae for Unity consumption


if __name__ == "__main__":
    cli()


