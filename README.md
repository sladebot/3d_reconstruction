## Monocular Video to 3D reconstruction

### Setup project

Run the following to setup the project locally or after cloning it in colab

```shell
bash scripts/setup.sh
```

Note: This works with CUDA 10.2 and Pytorch 1.9 and uses conda to install dependencies. 

### CLI

To use the cli please install it with `pip install -e .`, while in the root directory of the project.

After installation you'll be able to use it with the command `metacast`, and here's a list of options it takes: 

```shell

Usage: metacast [OPTIONS]

Options:
  -i, --input_video TEXT          Input MP4 file  [required]
  -o, --output_dir DIRECTORY      Directory to store OBJ files.  [required]
  -r, --resolution INTEGER        Resolution  [default: 256]
  -f, --fps INTEGER               Sampling FPS from video (This is not the
                                  input video fps)  [default: 24]
  -g, --gpu_id INTEGER            Sampling FPS from video (This is not the
                                  input video fps)  [default: 0]
  -c, --meshclean / --no-meshclean
                                  [default: no-meshclean]
  -p, --process / --no-process    [default: no-process]
  -h, --help                      Show this message and exit.  [default:
                                  False]
```

E.g. - 

```shell
metacast -i inputs/SkateBoarder.mp4 --output_dir=results -r 256  --fps 24 --gpu_id 0 --meshclean
```

This will do the following:

1. Convert the mp4 video into frames
2. Pre-process it to generate skeletal keypoints
3. Reconstruct the mesh from the sdf predicted by the network (Used PifuHD)
4. Save it to obj file

If `--meshclean` is provided it also does the following:

1. Uses open3d to cleanup the mesh
2. Computes the connected components and removes the outliers.

Here's an example:

![Refine](images/refine.png)


