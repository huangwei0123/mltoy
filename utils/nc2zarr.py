import xarray as xr
import zarr

INPUT_FILE = "data/era5-t2m-5deg.nc"
OUTPUT_ZARR = "data/era5-t2m-5deg.zarr"

ds_zarr = xr.open_dataset(
    INPUT_FILE,
    chunks={
        "valid_time": 128
    }
)

print("Writing Zarr...")
ds_zarr.to_zarr(OUTPUT_ZARR, mode="w")

ds_zarr = ds_zarr.chunk({
    "valid_time": 128,
    "latitude": 37,
    "longitude": 72
})

###################################################
# Save Zarr
###################################################

print("Writing Zarr...")

ds_zarr.to_zarr(
    OUTPUT_ZARR,
    mode="w"
)

print()
print("Done")
print("Zarr:", OUTPUT_ZARR)
