import logging
import click

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
    "--input",
    required=True,
    help="Input MP4 file",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(file_okay=False, writable=True),
    required=True,
    help="Directory to store OBJ files.",
)
def cli(input, output, generate_unity_data=True):
    ctx = click.get_current_context()
    # Convert to frames

    # Predict & Save series of obj
    # if generate_unity_data:
        # Convert to dae for Unity consumption


