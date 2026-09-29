import xarray as xr
import zarr

INPUT_FILE = "data/era5-t2m.nc"
OUTPUT_ZARR = "data/era5-t2m-5deg.zarr"

print("Opening dataset...")

ds = xr.open_dataset(
    INPUT_FILE,
    chunks={
        "valid_time": 64
    }
)

print(ds)

###################################################
# Kelvin -> Celsius
###################################################

ds["t2m"] = ds["t2m"] - 273.15

ds["t2m"].attrs["units"] = "degC"

###################################################
# Downsample from 0.25° to 5°
###################################################
#
# 5° / 0.25° = 20
#
###################################################

print("Coarsening latitude/longitude...")

ds_5deg = ds.coarsen(
    latitude=20,
    longitude=20,
    boundary="trim"
).mean()

print(ds_5deg)

###################################################
# Rechunk
###################################################

ds_5deg = ds_5deg.chunk({
    "valid_time": 128,
    "latitude": 37,
    "longitude": 72
})

###################################################
# Save
###################################################

print("Writing Zarr...")

ds_5deg.to_zarr(
    OUTPUT_ZARR,
    mode="w"
)

print()
print("Done")
print("Saved:", OUTPUT_ZARR)

