using Test
push!(LOAD_PATH, "../julia/")
using SST

y, m = SST.get_year_month_file("/data/SST/Global/monthly/2000/TERRA_MODIS.20000201_20000229.L3m.MO.SST4.sst4.9km.nc")
@test y == 2000
@test m == 2

sstclimfile = SST.get_monthly_clim_filename("TERRA", "MODIS", 2000, 2020, 2)
@test sstclimfile == "TERRA_MODIS.20000201_20200229.L3m.MC.SST4.sst4.9km.nc"
sstclimfile = SST.get_monthly_clim_filename("TERRA", "MODIS", 2000, 2020, 2, res="4km")
@test sstclimfile == "TERRA_MODIS.20000201_20200229.L3m.MC.SST4.sst4.4km.nc"

testfile = "/data/SST/Global/monthly/2000/TERRA_MODIS.20000201_20000229.L3m.MO.SST4.sst4.9km.nc"
@time lon, lat, sst = SST.read_sst_oceancolor_L3(testfile, [-20, -8., 25., 33.]);
@test length(lon) == 144
@test length(lat) == 96
@test lon[1] == -19.958334f0
@test lat[2] == 32.874996f0
@test sst[20, 30] == 18.15f0
