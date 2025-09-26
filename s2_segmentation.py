"""
Sentinel-2 semantic segmentation pipeline powered by the geoai-py library.

This script provides a simple end-to-end workflow:
- Download Sentinel-2 bands from Planetary Computer using a STAC item URL
- Create training chips from raster + labels
- Train a semantic segmentation model (e.g., U-Net) with segmentation-models-pytorch backend
- Run tiled inference on large rasters or batch over a folder of chips
- Optional: vectorize the mask to GeoJSON and add basic geometric properties

Requires: pip install geoai-py
Docs: https://opengeoai.org/
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional


def download_sentinel2(item_url: str, output_dir: str, merge_bands: bool = True, cell_size: int = 10) -> dict:
    """Download Sentinel-2 bands given a Planetary Computer STAC item URL.

    Returns a dict of band->path. If merge_bands, also includes merged GeoTIFF path under key "merged".
    """
    import geoai  # external package

    bands = [
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B09",
        "B11",
        "B12",
        "AOT",
        "WVP",
        "SCL",
    ]

    output_dir = str(Path(output_dir).expanduser().absolute())
    os.makedirs(output_dir, exist_ok=True)

    result = geoai.download_pc_stac_item(
        item_url=item_url,
        bands=bands,
        output_dir=output_dir,
        show_progress=True,
        merge_bands=merge_bands,
        merged_filename="sentinel2_all_bands.tif",
        overwrite=False,
        cell_size=cell_size,
    )
    return result


def chip_training_data(
    in_raster: str,
    in_vector: str,
    out_folder: str,
    tile_size: int = 512,
    stride: int = 256,
    buffer_radius: int = 0,
) -> List[str]:
    """Create image chips and binary masks from raster and vector labels."""
    import geoai

    out_folder = str(Path(out_folder).expanduser().absolute())
    os.makedirs(out_folder, exist_ok=True)

    tiles = geoai.export_geotiff_tiles(
        in_raster=in_raster,
        out_folder=out_folder,
        in_class_data=in_vector,
        tile_size=tile_size,
        stride=stride,
        buffer_radius=buffer_radius,
    )
    return tiles


def train(
    images_dir: str,
    labels_dir: str,
    output_dir: str,
    architecture: str = "unet",
    encoder_name: str = "resnet34",
    encoder_weights: Optional[str] = "imagenet",
    num_channels: int = 3,
    num_classes: int = 2,
    batch_size: int = 8,
    num_epochs: int = 50,
    learning_rate: float = 1e-3,
    val_split: float = 0.2,
    verbose: bool = True,
) -> str:
    """Train a segmentation model and return path to best_model.pth."""
    import geoai

    output_dir = str(Path(output_dir).expanduser().absolute())
    os.makedirs(output_dir, exist_ok=True)

    geoai.train_segmentation_model(
        images_dir=images_dir,
        labels_dir=labels_dir,
        output_dir=output_dir,
        architecture=architecture,
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        num_channels=num_channels,
        num_classes=num_classes,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        val_split=val_split,
        verbose=verbose,
    )

    best_model = str(Path(output_dir) / "best_model.pth")
    return best_model


def infer_raster(
    input_path: str,
    output_path: str,
    model_path: str,
    architecture: str = "unet",
    encoder_name: str = "resnet34",
    num_channels: int = 3,
    num_classes: int = 2,
    window_size: int = 512,
    overlap: int = 256,
    batch_size: int = 4,
) -> str:
    """Run tiled inference over a large raster and save mask GeoTIFF."""
    import geoai

    out_dir = str(Path(output_path).expanduser().absolute())
    Path(out_dir).parent.mkdir(parents=True, exist_ok=True)

    geoai.semantic_segmentation(
        input_path=input_path,
        output_path=output_path,
        model_path=model_path,
        architecture=architecture,
        encoder_name=encoder_name,
        num_channels=num_channels,
        num_classes=num_classes,
        window_size=window_size,
        overlap=overlap,
        batch_size=batch_size,
    )
    return output_path


def infer_folder(
    input_dir: str,
    output_dir: str,
    model_path: str,
    architecture: str = "unet",
    encoder_name: str = "resnet34",
    num_channels: int = 3,
    num_classes: int = 2,
    window_size: int = 512,
    overlap: int = 256,
    batch_size: int = 8,
    quiet: bool = True,
) -> str:
    """Run batch inference over a folder of chips and save mask GeoTIFFs."""
    import geoai

    output_dir = str(Path(output_dir).expanduser().absolute())
    os.makedirs(output_dir, exist_ok=True)

    geoai.semantic_segmentation_batch(
        input_dir=input_dir,
        output_dir=output_dir,
        model_path=model_path,
        architecture=architecture,
        encoder_name=encoder_name,
        num_channels=num_channels,
        num_classes=num_classes,
        window_size=window_size,
        overlap=overlap,
        batch_size=batch_size,
        quiet=quiet,
    )
    return output_dir


def vectorize_mask(
    mask_tif: str,
    out_geojson: str,
    epsilon: float = 2.0,
) -> str:
    """Vectorize a binary mask to polygons and save a GeoJSON with geometry properties."""
    import geoai

    Path(out_geojson).parent.mkdir(parents=True, exist_ok=True)
    gdf = geoai.orthogonalize(mask_tif, out_geojson, epsilon=epsilon)
    gdf = geoai.add_geometric_properties(gdf, area_unit="m2", length_unit="m")
    gdf.to_file(out_geojson, driver="GeoJSON")
    return out_geojson


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sentinel-2 segmentation with geoai-py")

    sub = p.add_subparsers(dest="cmd", required=True)

    p_dl = sub.add_parser("download", help="Download Sentinel-2 bands from a STAC item URL")
    p_dl.add_argument("item_url", help="Planetary Computer STAC item URL")
    p_dl.add_argument("--out", dest="output_dir", default="data/sentinel2", help="Output dir")
    p_dl.add_argument("--no-merge", action="store_true", help="Do not merge bands into one GeoTIFF")
    p_dl.add_argument("--cell-size", type=int, default=10, help="Resample bands to this cell size (m)")

    p_chip = sub.add_parser("chip", help="Create training chips from raster + vector labels")
    p_chip.add_argument("raster")
    p_chip.add_argument("vector")
    p_chip.add_argument("--out", dest="out_folder", default="data/tiles")
    p_chip.add_argument("--tile-size", type=int, default=512)
    p_chip.add_argument("--stride", type=int, default=256)
    p_chip.add_argument("--buffer", dest="buffer_radius", type=int, default=0)

    p_train = sub.add_parser("train", help="Train a semantic segmentation model")
    p_train.add_argument("images_dir")
    p_train.add_argument("labels_dir")
    p_train.add_argument("--out", dest="output_dir", default="models/unet")
    p_train.add_argument("--arch", dest="architecture", default="unet")
    p_train.add_argument("--encoder", dest="encoder_name", default="resnet34")
    p_train.add_argument("--encoder-weights", default="imagenet")
    p_train.add_argument("--channels", dest="num_channels", type=int, default=3)
    p_train.add_argument("--classes", dest="num_classes", type=int, default=2)
    p_train.add_argument("--batch", dest="batch_size", type=int, default=8)
    p_train.add_argument("--epochs", dest="num_epochs", type=int, default=50)
    p_train.add_argument("--lr", dest="learning_rate", type=float, default=1e-3)
    p_train.add_argument("--val-split", dest="val_split", type=float, default=0.2)
    p_train.add_argument("--quiet", dest="verbose", action="store_false")

    p_infer = sub.add_parser("infer", help="Run tiled inference on a raster")
    p_infer.add_argument("input")
    p_infer.add_argument("model")
    p_infer.add_argument("--out", dest="output", default="outputs/mask.tif")
    p_infer.add_argument("--arch", dest="architecture", default="unet")
    p_infer.add_argument("--encoder", dest="encoder_name", default="resnet34")
    p_infer.add_argument("--channels", dest="num_channels", type=int, default=3)
    p_infer.add_argument("--classes", dest="num_classes", type=int, default=2)
    p_infer.add_argument("--window", dest="window_size", type=int, default=512)
    p_infer.add_argument("--overlap", type=int, default=256)
    p_infer.add_argument("--batch", dest="batch_size", type=int, default=4)

    p_batch = sub.add_parser("infer-batch", help="Run batch inference over a folder of chips")
    p_batch.add_argument("input_dir")
    p_batch.add_argument("model")
    p_batch.add_argument("--out", dest="output_dir", default="outputs/predictions")
    p_batch.add_argument("--arch", dest="architecture", default="unet")
    p_batch.add_argument("--encoder", dest="encoder_name", default="resnet34")
    p_batch.add_argument("--channels", dest="num_channels", type=int, default=3)
    p_batch.add_argument("--classes", dest="num_classes", type=int, default=2)
    p_batch.add_argument("--window", dest="window_size", type=int, default=512)
    p_batch.add_argument("--overlap", type=int, default=256)
    p_batch.add_argument("--batch", dest="batch_size", type=int, default=8)
    p_batch.add_argument("--loud", dest="quiet", action="store_false")

    p_vec = sub.add_parser("vectorize", help="Vectorize a predicted mask GeoTIFF to GeoJSON")
    p_vec.add_argument("mask")
    p_vec.add_argument("--out", dest="out_geojson", default="outputs/mask.geojson")
    p_vec.add_argument("--epsilon", type=float, default=2.0)

    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)

    if args.cmd == "download":
        res = download_sentinel2(
            item_url=args.item_url,
            output_dir=args.output_dir,
            merge_bands=not args.no_merge,
            cell_size=args.cell_size,
        )
        print("Downloaded:")
        for k, v in res.items():
            print(f"  {k}: {v}")

    elif args.cmd == "chip":
        tiles = chip_training_data(
            in_raster=args.raster,
            in_vector=args.vector,
            out_folder=args.out_folder,
            tile_size=args.tile_size,
            stride=args.stride,
            buffer_radius=args.buffer_radius,
        )
        print(f"Created {len(tiles)} tiles in {args.out_folder}")

    elif args.cmd == "train":
        model_path = train(
            images_dir=args.images_dir,
            labels_dir=args.labels_dir,
            output_dir=args.output_dir,
            architecture=args.architecture,
            encoder_name=args.encoder_name,
            encoder_weights=args.encoder_weights,
            num_channels=args.num_channels,
            num_classes=args.num_classes,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            val_split=args.val_split,
            verbose=args.verbose,
        )
        print(f"Best model saved to: {model_path}")

    elif args.cmd == "infer":
        out = infer_raster(
            input_path=args.input,
            output_path=args.output,
            model_path=args.model,
            architecture=args.architecture,
            encoder_name=args.encoder_name,
            num_channels=args.num_channels,
            num_classes=args.num_classes,
            window_size=args.window_size,
            overlap=args.overlap,
            batch_size=args.batch_size,
        )
        print(f"Saved mask: {out}")

    elif args.cmd == "infer-batch":
        out = infer_folder(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            model_path=args.model,
            architecture=args.architecture,
            encoder_name=args.encoder_name,
            num_channels=args.num_channels,
            num_classes=args.num_classes,
            window_size=args.window_size,
            overlap=args.overlap,
            batch_size=args.batch_size,
            quiet=args.quiet,
        )
        print(f"Saved predictions to: {out}")

    elif args.cmd == "vectorize":
        out = vectorize_mask(
            mask_tif=args.mask,
            out_geojson=args.out_geojson,
            epsilon=args.epsilon,
        )
        print(f"Saved GeoJSON: {out}")


if __name__ == "__main__":
    main()
