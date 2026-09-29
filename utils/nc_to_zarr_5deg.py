import xarray as xr
import zarr

INPUT_FILE = "data/era5-t2m.nc"

OUTPUT_NETCDF = "data/era5-t2m-5deg.nc"
OUTPUT_ZARR = "data/era5-t2m-5deg.zarr"

print("Opening dataset...")

# ds = xr.open_dataset(
#     INPUT_FILE,
#     chunks={
#         "valid_time": 64
#     }
# )

ds = xr.open_dataset(INPUT_FILE)
print(ds)

###################################################
# Kelvin -> Celsius
###################################################

ds["t2m"] = ds["t2m"] - 273.15
ds["t2m"].attrs["units"] = "degC"

ds = ds.chunk({"valid_time": 64})

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
# Save NetCDF first
###################################################

print("Writing NetCDF...")

# ds_5deg.to_netcdf(
#     OUTPUT_NETCDF,
#     engine="netcdf4"
# )

ds_5deg = ds_5deg.chunk({ "valid_time": 128, "latitude": 37, "longitude": 72 })

encoding = {
    "t2m": {
        "zlib": True,
        "complevel": 4,
        "shuffle": True
    }
}

ds_5deg.to_netcdf(OUTPUT_NETCDF, engine="netcdf4", encoding=encoding)

print("Saved:", OUTPUT_NETCDF)

###################################################
# Re-open and rechunk for Zarr
###################################################

# print("Re-opening 5 degree NetCDF...")

# ds_zarr = xr.open_dataset(
#     OUTPUT_NETCDF,
#     chunks={
#         "valid_time": 128
#     }
# )

print("Rechunking for Zarr...")
ds_zarr = ds_5deg.chunk({
    "valid_time": 128,
    "latitude": 37,
    "longitude": 72
})

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
print("NetCDF:", OUTPUT_NETCDF)
print("Zarr:", OUTPUT_ZARR)
