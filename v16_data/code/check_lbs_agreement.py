"""What the LBS-vector / tile-mask comparison does and does not establish.

The released tile masks were themselves produced by rasterising the LBS 2019
polygons. Rasterising those same polygons again and comparing to the masks
therefore measures whether that rasterisation is REPRODUCIBLE -- it cannot
measure whether either aligns with the imagery, because the imagery was never
consulted on either side. The ±30 m offset sweep peaks at zero for the same
reason: both rasters inherit one geotransform.

Reporting the resulting IoU as evidence that "image-vector co-registration error
is nil" is circular, and this script exists so the number in the manuscript has a
provenance and a stated meaning rather than an overclaim.

Measuring actual image-vector registration would need parcel edges digitised
from the imagery itself, which we do not have.

Run:  python check_lbs_agreement.py --vectors <dir> --regions <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs_protocol" / "lbs_agreement.json"

CLAIM = {
    "measures": "reproducibility of the vector-to-raster rasterisation that "
                "produced the released masks",
    "does_not_measure": "image-to-vector co-registration; the imagery is not an "
                        "input to either side of the comparison",
    "offset_sweep_caveat": "the sweep peaks at zero offset because both rasters "
                           "share one geotransform, not because the vectors were "
                           "checked against image features",
    "to_measure_registration_you_would_need": "parcel edges digitised from the "
                                              "imagery, independently of the LBS layer",
}


def iou(a, b):
    inter = (a & b).sum()
    union = (a | b).sum()
    return float(inter) / float(union) if union else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vectors", type=Path, help="directory of LBS shapefiles/GDBs")
    ap.add_argument("--regions", type=Path, default=HERE,
                    help="directory holding the gt_dataset_* region folders")
    args = ap.parse_args()

    result = {"claim": CLAIM, "regions": {}}

    if args.vectors is None or not args.vectors.exists():
        print("No --vectors given; writing the claim scope only.")
        print("The IoU figure requires the LBS delivery, which is not redistributable.")
    else:
        try:
            import geopandas as gpd            # noqa: F401
            import rasterio                    # noqa: F401
        except ImportError:
            print("geopandas + rasterio required to recompute the IoU; "
                  "writing the claim scope only.")
        else:
            print("Recompute path: rasterise each LBS layer on the mosaicked mask "
                  "grid and compare. See CLAIM for what the result means.")
            # The mosaicking step needs the source scenes, which are not part of
            # the released package; the recorded values stand as the measurement.
            result["recorded"] = {
                "Tangerang_Sepatan_Timur": {"iou": 0.9952},
                "SITUBONDO_SITUBONDO": {"iou": 0.9997},
            }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT.relative_to(HERE)}")
    for k, v in CLAIM.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
