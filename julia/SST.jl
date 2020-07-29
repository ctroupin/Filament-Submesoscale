module SST

using NCDatasets
using Statistics
using Dates

"""
    get_sst_file_list(datadir)

Return a list of files located in `datadir`.

## Example
```julia-repl
julia> filelist = get_sst_file_list(datadir)
```
"""
function get_sst_file_list(datadir::String, sat::String="TERRA", res::String="9km")::Array
    filelist = []
    for (root, dirs, files) in walkdir(datadir)
        for file in files
            if endswith(file, "$(res).nc") & startswith(file, sat)
                push!(filelist, joinpath(root, file))
            end
        end
    end
    @info("Found $(length(filelist)) files")
    return filelist
end

"""
    get_year_month_file(filename)

Return the year and the month corresponding to the netCDF file `filename`

## Example
```julia-repl
julia>
```
"""
function get_year_month_file(filename::String)
    NCDatasets.Dataset(filename) do nc
        tstart = nc.attrib["time_coverage_start"]
        tend = nc.attrib["time_coverage_end"]
        dateformat = Dates.DateFormat("yyyy-mm-ddTHH:MM:SS.000Z");
        datestart = Dates.DateTime(tstart, dateformat)
        dateend = Dates.DateTime(tend, dateformat)
        datemean = datestart + Millisecond(0.5 * (dateend - datestart).value)

        return year(datemean), month(datemean)
    end
end

"""
    get_monthly_clim_filename(sat, sensor, yearstart, yearend, month)

Return the file name of the climatological file according to the satellite,
mission and the date

## Example
```julia-repl
sstclimfile = get_monthly_clim_filename("TERRA", "MODIS", 2000, 2020, 2)
> "TERRA_MODIS.20000201_20200229.L3m.MC.SST4.sst4.9km.nc"
```

"""
function get_monthly_clim_filename(sat::String, sensor::String, yearstart::Int64, yearend::Int64,
        month::Int64; res::String="9km")

    mm = lpad(string(month), 2, "0")
    numdays = Dates.daysinmonth(Date(yearend, month))
    ddend = lpad(string(numdays), 2, "0")
    fname = "$(sat)_$(sensor).$(yearstart)$(mm)01_$(yearend)$(mm)$(ddend).L3m.MC.SST4.sst4.$(res).nc"

    return fname
end

"""
    read_sst_oceancolor_L3(datafile, domain)

Extrat the SST and the coordinates from the netCDF `datafile` in the region
defined by `domain` (lonmin, lonmax, latmin, latmax).

## Example
```julia-repl
lon, lat, sst = read_sst_oceancolor_L3(datafile, [-20, -8., 25., 33.])
```
"""
function read_sst_oceancolor_L3(datafile::String, domain::Array)

    NCDatasets.Dataset(datafile) do nc
        lon = nc["lon"][:]
        lat = nc["lat"][:]
        goodlon = (lon .< domain[2]) .& (lon .> domain[1])
        goodlat = (lat .< domain[4]) .& (lat .> domain[3])
        sst = nc["sst4"][goodlon, goodlat]
        return coalesce.(lon[goodlon], NaN), coalesce.(lat[goodlat], NaN), coalesce.(sst, NaN)
    end
end

"""
    create_sst_file(filename, lons, lats, times, sstanom)

Write the coordinates `lons`, `lates`, the times `times` and the SST anomalies
`sstanom` in the netCDF file `filename`.

## Example

"""
function create_sst_file(filename::String, lons, lats, times, sstanom, mask; valex=-999.9)
    Dataset(filename, "c") do ds

        # Dimensions
        ds.dim["lon"] = length(lons)
        ds.dim["lat"] = length(lats)
        ds.dim["time"] = Inf # unlimited dimension

        # Declare variables
        ncsst = defVar(ds,"sstanom", Float64, ("lon", "lat", "time"))
        ncsst.attrib["missing_value"] = Float64(valex)
        ncsst.attrib["standard_name"] = "sea_water_temperature_anomaly"
        ncsst.attrib["_FillValue"] = Float64(valex)
        ncsst.attrib["units"] = "degree_C"

        ncmask = defVar(ds,"mask", Int64, ("lon", "lat"))
        ncmask.attrib["long_name"] = "land sea mask"

        nctime = defVar(ds,"time", Float32, ("time",))
        nctime.attrib["missing_value"] = Float32(valex)
        nctime.attrib["units"] = "seconds since 1981-01-01 00:00:00"
        nctime.attrib["time"] = "time"

        nclon = defVar(ds,"lon", Float32, ("lon",))
        nclon.attrib["missing_value"] = Float32(valex)
        nclon.attrib["_FillValue"] = Float32(valex)
        nclon.attrib["units"] = "degrees East"
        nclon.attrib["lon"] = "longitude"

        nclat = defVar(ds,"lat", Float32, ("lat",))
        nclat.attrib["missing_value"] = Float32(valex)
        nclat.attrib["_FillValue"] = Float32(valex)
        nclat.attrib["units"] = "degrees North"
        nclat.attrib["lat"] = "latitude"

        # Global attributes
        ds.attrib["institution"] = "GHER - University of Liège"
        ds.attrib["title"] = "SST anomalies from MODIS TERRA 4µm"
        ds.attrib["comment"] = "Original data by Ocean Color"
        ds.attrib["data URL"] = "https://oceandata.sci.gsfc.nasa.gov/MODIS-Terra/Mapped/Monthly/9km/"
        ds.attrib["author"] = "C. Troupin, ctroupin@uliege"
        ds.attrib["tool"] = "create_nc_tile"
        ds.attrib["institution_url"] = "http://modb.oce.ulg.ac.be/"

        # Define variables
        ncmask[:] = mask
        ncsst[:] = sstanom
        nctime[:] = times
        nclon[:] = lons
        nclat[:] = lats;

    end
end;

end
